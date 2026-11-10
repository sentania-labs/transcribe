"""transcribe: CPU speech-to-text with speaker labels, OpenAI-compatible.

POST /v1/audio/transcriptions (multipart, the OpenAI shape). Two passes over
the whole recording, both on the CPU:

  1. pyannote speaker-diarization-community-1 labels who speaks when;
  2. faster-whisper (CTranslate2, int8) transcribes the recording once, with
     word timestamps and its VAD filter.

Each word then goes to the speaker whose turn overlaps it most, and runs of
one speaker's words become segments. Transcribing turn by turn (the GPU
version on the Spark did that) costs a full 30-second encoder pass for every
short turn, which on a CPU is several times the work of one pass over the
recording.

response_format:
  json (default)  {"text": "Speaker 1: ...\\nSpeaker 2: ...", "usage": ...}
  text            the same text, plain
  verbose_json    task, language, duration, text, segments [{id, speaker, start, end, text}]
  diarized_json   task, duration, text, segments [{type, id, speaker, start, end, text}], usage
Optional form fields: language, prompt (Whisper initial prompt), num_speakers,
min_speakers, max_speakers (passed to pyannote).

Long recordings: POST /v1/audio/transcriptions/jobs takes the same form and
answers 202 {"id", "status"} at once; GET /v1/audio/transcriptions/jobs/{id}
returns the status and, when done, the result in the requested format. Jobs
live in memory for JOB_TTL seconds and are lost on restart.

Auth: TRANSCRIBE_API_KEY as a bearer token, checked before any of the upload
is read. Limits: MAX_UPLOAD_BYTES per upload, MAX_SECONDS of audio, MAX_JOBS
jobs at once (more get 429). A synchronous request whose client goes away is
cancelled at the next progress step.
"""
import asyncio
import hmac
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from app.assign import assign_speakers, cpu_limit, windows

log = logging.getLogger("transcribe")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

API_KEY = os.environ["TRANSCRIBE_API_KEY"]
PIPELINE_PATH = os.environ.get("DIARIZATION_PIPELINE", "/models/pyannote/speaker-diarization-community-1")
WHISPER_PATH = os.environ.get("WHISPER_MODEL", "/models/whisper/large-v3-turbo")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", 5))
# pyannote slides a 10 s window by this fraction of it. The speaker embedding
# for every window is most of the CPU time, so 0.1 (pyannote's default) costs
# about 2.6 times what 0.25 does on a CPU (measured; 0.5 fragments turns).
DIARIZATION_STEP = float(os.environ.get("DIARIZATION_STEP", 0.25))
THREADS = int(os.environ.get("THREADS") or cpu_limit())
WHISPER_WINDOW = float(os.environ.get("WHISPER_WINDOW", 1200))
SAMPLE_RATE = 16000
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", 2.5 * 3600))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 1024**3))
MAX_JOBS = int(os.environ.get("MAX_JOBS", 1))
JOB_TTL = float(os.environ.get("JOB_TTL", 24 * 3600))
FORMATS = ("json", "text", "verbose_json", "diarized_json")
# Containers ffmpeg may read. Playlists and concat lists (hls, concat, ...) are
# refused: they make ffmpeg open other files.
AUDIO_FORMATS = {"mp3", "wav", "flac", "ogg", "matroska", "webm", "mov", "mp4", "m4a",
                 "3gp", "3g2", "mj2", "aac", "amr", "aiff", "caf", "w64", "opus"}

app = FastAPI(title="transcribe", docs_url=None, redoc_url=None, openapi_url=None)
models: dict = {}
model_error = "loading"
cpu_lock = threading.Lock()   # one job on the CPU at a time, whatever MAX_JOBS says
jobs = asyncio.Semaphore(MAX_JOBS)
async_jobs: dict[str, dict] = {}


class Cancelled(Exception):
    pass


def load_models():
    """Load both models, retrying until their files exist (the volume is seeded by hand)."""
    global model_error
    while True:
        missing = [p for p in (os.path.join(PIPELINE_PATH, "config.yaml"),
                               os.path.join(WHISPER_PATH, "model.bin")) if not os.path.exists(p)]
        if missing:
            model_error = f"model files missing: {', '.join(missing)}"
            log.warning("%s; retrying in 30 s", model_error)
            time.sleep(30)
            continue
        try:
            import torch
            from faster_whisper import WhisperModel
            from pyannote.audio import Pipeline
            torch.set_num_threads(THREADS)
            pipeline = Pipeline.from_pretrained(PIPELINE_PATH)
            pipeline.to(torch.device("cpu"))
            pipeline._segmentation.step = DIARIZATION_STEP * pipeline._segmentation.duration
            whisper = WhisperModel(WHISPER_PATH, device="cpu", compute_type=COMPUTE_TYPE,
                                   cpu_threads=THREADS, num_workers=1)
        except Exception as e:  # keep serving /health with the reason
            model_error = f"model load failed: {e!r}"[:500]
            log.exception("model load failed; retrying in 60 s")
            time.sleep(60)
            continue
        models.update(pipeline=pipeline, whisper=whisper)
        model_error = ""
        log.info("models loaded: threads=%d compute_type=%s beam=%d diarization_step=%s",
                 THREADS, COMPUTE_TYPE, BEAM_SIZE, DIARIZATION_STEP)
        return


@app.on_event("startup")
async def startup():
    threading.Thread(target=load_models, daemon=True).start()
    app.state.reaper = asyncio.create_task(reap_jobs())


class Gate:
    """Plain ASGI middleware: refuse before a byte of the body is read.

    Not @app.middleware("http"): that wrapper hides the client's disconnect
    from request.is_disconnected(), and a cancelled request would then hold
    every CPU until its job ended. Checked here, in order: the key (401); for
    an upload, a Content-Length (411: without one the size is unknown until
    the body has been spooled to disk) within MAX_UPLOAD_BYTES (413), models
    loaded (503), a free job slot (429).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        token = headers.get("authorization", "").removeprefix("Bearer ").strip()
        status, message = 0, ""
        if not hmac.compare_digest(token.encode(), API_KEY.encode()):
            status, message = 401, "invalid api key"
        elif scope["method"] == "POST" and scope["path"].startswith("/v1/audio/transcriptions"):
            length = headers.get("content-length", "")
            if not length.isdigit():
                status, message = 411, "Content-Length required"
            elif int(length) > MAX_UPLOAD_BYTES:
                status, message = 413, "upload too large"
            elif model_error:
                status, message = 503, model_error
            elif jobs.locked():
                status, message = 429, f"{MAX_JOBS} transcriptions already running, retry later"
        if status:
            return await JSONResponse({"error": {"message": message}}, status_code=status)(scope, receive, send)
        return await self.app(scope, receive, send)


app.add_middleware(Gate)


def decode(fd: int) -> np.ndarray:
    """Any audio ffmpeg reads -> 16 kHz mono float32, at most MAX_SECONDS.

    fd is the upload Starlette already spooled to disk. ffprobe and ffmpeg open
    it as /dev/fd/N (a fresh, seekable open of the same file), so the upload is
    never copied a second time.
    """
    path = f"/dev/fd/{fd}"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "json", path],
        capture_output=True, check=False, pass_fds=(fd,))
    try:
        names = set(json.loads(probe.stdout)["format"]["format_name"].split(","))
    except (ValueError, KeyError):
        raise HTTPException(status_code=400, detail="not an audio file ffmpeg can read")
    if not names & AUDIO_FORMATS:
        raise HTTPException(status_code=400, detail=f"unsupported container {','.join(sorted(names))}")
    # Decoded to a file, then read once: through a pipe the samples would sit in
    # memory twice (the bytes, then a writable copy for torch). 2 h = 460 MB.
    with tempfile.NamedTemporaryFile(suffix=".f32") as raw:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-t", str(MAX_SECONDS + 1),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", raw.name],
            capture_output=True, check=False, pass_fds=(fd,))
        if proc.returncode != 0:
            raise HTTPException(status_code=400, detail="could not decode audio: "
                                + proc.stderr.decode(errors="replace")[-300:])
        audio = np.fromfile(raw.name, dtype=np.float32)
    if not len(audio):
        raise HTTPException(status_code=400, detail="no audio in the file")
    if len(audio) / SAMPLE_RATE > MAX_SECONDS:
        raise HTTPException(status_code=413, detail=f"audio longer than {MAX_SECONDS:.0f} s")
    return audio


def diarize(audio: np.ndarray, cancel: threading.Event, **speakers) -> list[tuple[float, float, str]]:
    import torch

    def hook(*args, **kwargs):  # pyannote calls this between and within steps
        if cancel.is_set():
            raise Cancelled()

    wav = torch.from_numpy(audio).unsqueeze(0)
    out = models["pipeline"]({"waveform": wav, "sample_rate": SAMPLE_RATE}, hook=hook, **speakers)
    # community-1 also returns an overlap-free version: every instant has at
    # most one speaker, which is what assigning each word to one speaker needs.
    ann = getattr(out, "exclusive_speaker_diarization", None)
    if ann is None:
        ann = getattr(out, "speaker_diarization", out)
    return [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]


def transcribe_words(audio: np.ndarray, cancel: threading.Event, language, prompt):
    """-> ([(start, end, word)], detected language).

    Whisper gets the recording in windows of WHISPER_WINDOW seconds cut at the
    quietest point near each boundary: faster-whisper computes the log-mel
    features for everything it is given at once (about 3 GB per hour of audio),
    so whole-file input would make memory grow with the recording.
    """
    words, detected = [], language
    for start, end in windows(audio, SAMPLE_RATE, WHISPER_WINDOW):
        segments, info = models["whisper"].transcribe(
            audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)],
            language=detected or None, initial_prompt=prompt or None, beam_size=BEAM_SIZE,
            word_timestamps=True, vad_filter=True,
            # Off: on hour-long audio, conditioning on the previous window is what
            # lets one hallucinated line repeat for minutes.
            condition_on_previous_text=False)
        for seg in segments:  # a generator: the work happens here, 30 s at a time
            if cancel.is_set():
                raise Cancelled()
            for w in seg.words or []:
                words.append((start + w.start, start + w.end, w.word))
        # Detected once, from the first window, then held: a later window of
        # silence or crosstalk must not switch the language mid-meeting.
        detected = detected or info.language
    return words, detected


def run_job(audio: np.ndarray, cancel: threading.Event, language, prompt, speakers) -> dict:
    """The CPU part of a job. Returns everything the formatter needs."""
    with cpu_lock:
        t0 = time.monotonic()
        turns = diarize(audio, cancel, **speakers)
        t1 = time.monotonic()
        words, detected = transcribe_words(audio, cancel, language, prompt)
        t2 = time.monotonic()
    duration = len(audio) / SAMPLE_RATE
    log.info("job done: audio %.0f s, diarize %.0f s, transcribe %.0f s, %d turns, %d words",
             duration, t1 - t0, t2 - t1, len(turns), len(words))
    return {"segments": assign_speakers(words, turns), "duration": duration,
            "language": language or detected}


def render(result: dict, response_format: str):
    segments, duration = result["segments"], result["duration"]
    labelled = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)
    usage = {"type": "duration", "seconds": int(round(duration))}
    if response_format == "text":
        return labelled
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": result["language"], "duration": round(duration, 2),
                "text": labelled, "segments": segments, "usage": usage}
    if response_format == "diarized_json":
        return {"task": "transcribe", "duration": round(duration, 2), "text": labelled,
                "segments": [{"type": "transcript.text.segment", **seg, "id": f"seg_{seg['id']}"}
                             for seg in segments],
                "usage": usage}
    return {"text": labelled, "usage": usage}


def check_request(response_format):
    """Re-checked here because Gate's checks and the job slot are not atomic."""
    if response_format not in FORMATS:
        raise HTTPException(status_code=400, detail=f"unsupported response_format {response_format!r}")
    if model_error:
        raise HTTPException(status_code=503, detail=model_error)
    if jobs.locked():
        raise HTTPException(status_code=429, detail=f"{MAX_JOBS} transcriptions already running, retry later")


async def receive(file: UploadFile) -> np.ndarray:
    """Decode the spooled upload, then delete the spool (it would sit on disk for the whole job)."""
    try:
        spool = file.file
        if hasattr(spool, "rollover"):
            spool.rollover()  # a small upload may still be in memory; give it a file
        spool.flush()
        if os.fstat(spool.fileno()).st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="upload too large")
        return await asyncio.to_thread(decode, spool.fileno())
    finally:
        await file.close()


def speaker_args(num_speakers, min_speakers, max_speakers):
    return {k: v for k, v in (("num_speakers", num_speakers), ("min_speakers", min_speakers),
                              ("max_speakers", max_speakers)) if v}


@app.get("/health")
async def health():
    if model_error:
        return JSONResponse({"status": "unavailable", "detail": model_error}, status_code=503)
    return {"status": "ok", "threads": THREADS, "busy": jobs.locked()}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("transcribe"),
    response_format: str = Form("json"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    check_request(response_format)
    async with jobs:
        audio = await receive(file)
        cancel = threading.Event()

        async def watch():
            while not cancel.is_set():
                if await request.is_disconnected():
                    log.info("client went away; cancelling")
                    cancel.set()
                await asyncio.sleep(2)

        watcher = asyncio.create_task(watch())
        try:
            result = await asyncio.to_thread(run_job, audio, cancel, language, prompt,
                                             speaker_args(num_speakers, min_speakers, max_speakers))
        except Cancelled:
            raise HTTPException(status_code=499, detail="client went away")
        finally:
            cancel.set()
            watcher.cancel()
    body = render(result, response_format)
    return PlainTextResponse(body) if response_format == "text" else JSONResponse(body)


@app.post("/v1/audio/transcriptions/jobs", status_code=202)
async def submit_job(
    file: UploadFile = File(...),
    model: str = Form("transcribe"),
    response_format: str = Form("json"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
):
    check_request(response_format)
    await jobs.acquire()
    try:
        audio = await receive(file)
    except BaseException:
        jobs.release()
        raise
    job_id = uuid.uuid4().hex
    job = async_jobs[job_id] = {"id": job_id, "status": "running", "created": time.time(),
                                "response_format": response_format}

    async def work():
        try:
            result = await asyncio.to_thread(run_job, audio, threading.Event(), language, prompt,
                                             speaker_args(num_speakers, min_speakers, max_speakers))
            job.update(status="completed", result=render(result, response_format))
        except Exception as e:
            log.exception("job %s failed", job_id)
            job.update(status="failed", error=getattr(e, "detail", None) or repr(e)[:300])
        finally:
            job["finished"] = time.time()
            jobs.release()

    job["_task"] = asyncio.create_task(work())  # keep a reference: the loop holds tasks weakly
    return {"id": job_id, "status": "running"}


@app.get("/v1/audio/transcriptions/jobs/{job_id}")
async def get_job(job_id: str):
    job = async_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job (finished jobs expire, and a restart forgets all)")
    out = {k: job[k] for k in ("id", "status", "created") if k in job}
    if job["status"] == "failed":
        out["error"] = job["error"]
    if job["status"] == "completed":
        if job["response_format"] == "text":
            out["text"] = job["result"]
        else:
            out["result"] = job["result"]
    return out


async def reap_jobs():
    while True:
        now = time.time()
        for job_id in [k for k, j in async_jobs.items() if now - j.get("finished", now) > JOB_TTL]:
            async_jobs.pop(job_id, None)
        await asyncio.sleep(300)
