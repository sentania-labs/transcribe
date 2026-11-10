"""transcribe: CPU speech-to-text with speaker labels and optional voice signatures.

POST /v1/audio/transcriptions (multipart, the OpenAI shape). Passes over the
recording, all on the CPU, one job at a time:

  1. pyannote speaker-diarization-community-1 labels who speaks when;
  2. faster-whisper (CTranslate2, int8) transcribes it in windows, with word
     timestamps and its VAD filter; each sentence goes to the speaker who
     covers most of its words (app/assign.py);
  3. only when include_speaker_embeddings=true: one voice signature per
     speaker and configured voice model, from clean speech (app/voice.py).

Formats: json, text, verbose_json, diarized_json. Long recordings go to the job
path (POST /v1/audio/transcriptions/jobs, GET .../jobs/{id}); its result is the
same JSON as the direct call. Every error is {"error": {"reason", "message"}}
with a stable reason (README.md). GET /v1/status reports models, limits and
whether a job is running.

Privacy: no audio, transcript, prompt or vector is ever logged or kept,
except a finished job's result for JOB_TTL seconds in memory. Logs carry job
ids, durations, sizes and reasons only.
"""
import asyncio
import hmac
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.assign import assign_speakers, cpu_limit, windows
from app.voice import VoiceConfigError, VoiceModel, clean_regions, file_sha256, parse_voice_models

log = logging.getLogger("transcribe")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Library loggers stay at WARNING: none of them is allowed to put text in the log.
for name in ("faster_whisper", "pyannote", "speechbrain", "lightning", "pytorch_lightning"):
    logging.getLogger(name).setLevel(logging.WARNING)

API_KEY = os.environ["TRANSCRIBE_API_KEY"]
PIPELINE_PATH = os.environ.get("DIARIZATION_PIPELINE", "/models/pyannote/speaker-diarization-community-1")
PIPELINE_ID = os.environ.get("DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-community-1")
PIPELINE_REVISION = os.environ.get("DIARIZATION_MODEL_REVISION", "")
WHISPER_PATH = os.environ.get("WHISPER_MODEL", "/models/whisper/large-v3-turbo")
WHISPER_ID = os.environ.get("WHISPER_MODEL_ID", "large-v3-turbo")
WHISPER_REVISION = os.environ.get("WHISPER_MODEL_REVISION", "")
WHISPER_SHA256 = os.environ.get("WHISPER_MODEL_SHA256", "")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", 5))
# pyannote slides a 10 s window by this fraction of it. The speaker embedding
# for every window is most of the CPU time, so 0.1 (pyannote's default) costs
# about 2.6 times what 0.25 does on a CPU (measured; 0.5 fragments turns).
DIARIZATION_STEP = float(os.environ.get("DIARIZATION_STEP", 0.25))
THREADS = int(os.environ.get("THREADS") or cpu_limit())
WHISPER_WINDOW = float(os.environ.get("WHISPER_WINDOW", 1200))
VOICE_MODELS_RAW = os.environ.get("VOICE_MODELS", "")
VOICE_MIN_SECONDS = float(os.environ.get("VOICE_MIN_SECONDS", 30))
SAMPLE_RATE = 16000
# Job path limit, and the direct path's (a synchronous call has to finish
# inside the ingress's 3600 s; at ~0.41 x audio length that is about 2h20m).
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", 2.5 * 3600))
DIRECT_MAX_SECONDS = min(MAX_SECONDS, float(os.environ.get("DIRECT_MAX_SECONDS", 8400)))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 1024**3))
MAX_JOBS = int(os.environ.get("MAX_JOBS", 1))
JOB_TTL = float(os.environ.get("JOB_TTL", 2 * 3600))
# Seconds of work per second of audio, for Retry-After (measured 0.41 on 4 CPUs).
WORK_RATIO = float(os.environ.get("WORK_RATIO", 0.45))
FORMATS = ("json", "text", "verbose_json", "diarized_json")
IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
# Containers ffmpeg may read. Playlists and concat lists (hls, concat, ...) are
# refused: they make ffmpeg open other files.
AUDIO_FORMATS = {"mp3", "wav", "flac", "ogg", "matroska", "webm", "mov", "mp4", "m4a",
                 "3gp", "3g2", "mj2", "aac", "amr", "aiff", "caf", "w64", "opus"}

app = FastAPI(title="transcribe", docs_url=None, redoc_url=None, openapi_url=None)
models: dict = {}
model_error = "loading"
model_info: dict = {}
cpu_lock = threading.Lock()   # one job on the CPU at a time, whatever MAX_JOBS says
jobs = asyncio.Semaphore(MAX_JOBS)
current: dict = {}            # the running job: started (monotonic), audio_seconds
async_jobs: dict[str, dict] = {}
idempotency: dict[str, str] = {}   # caller key -> job id, purged with the job


class Cancelled(Exception):
    pass


class ApiError(Exception):
    """An error with a stable reason code. The message never carries audio-derived text."""

    def __init__(self, status: int, reason: str, message: str, headers: dict | None = None):
        super().__init__(message)
        self.status, self.reason, self.message, self.headers = status, reason, message, headers or {}

    def body(self):
        return {"error": {"reason": self.reason, "message": self.message}}

    def response(self):
        return JSONResponse(self.body(), status_code=self.status, headers=self.headers)


def retry_after() -> int:
    """Seconds until the running job should finish (estimate), at least 30."""
    if not current:
        return 30
    remaining = current["audio_seconds"] * WORK_RATIO - (time.monotonic() - current["started"])
    return int(min(3600, max(30, remaining)))


def busy() -> ApiError:
    return ApiError(429, "busy", "a transcription is already running; retry later",
                    {"Retry-After": str(retry_after())})


@app.exception_handler(ApiError)
async def api_error(request, exc: ApiError):
    return exc.response()


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    reason = {404: "not_found", 405: "invalid_request"}.get(exc.status_code, "invalid_request")
    return ApiError(exc.status_code, reason, str(exc.detail)).response()


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Field names only: the default handler echoes the submitted values back.
    fields = sorted({".".join(str(p) for p in e.get("loc", [])[1:]) for e in exc.errors()})
    return ApiError(400, "invalid_request", f"invalid or missing fields: {', '.join(fields)}").response()


@app.exception_handler(Exception)
async def internal_error(request, exc):
    log.error("internal error on %s: %s", request.url.path, type(exc).__name__)
    return ApiError(500, "internal", "internal error").response()


def load_models():
    """Load every model, retrying until their files exist (the volume is seeded by hand)."""
    global model_error
    try:
        voice_entries = parse_voice_models(VOICE_MODELS_RAW)
    except VoiceConfigError as e:
        model_error = f"voice model config: {e}"
        log.error("%s", model_error)
        return
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
            whisper_sha = file_sha256(os.path.join(WHISPER_PATH, "model.bin"))
            if WHISPER_SHA256 and whisper_sha != WHISPER_SHA256:
                raise VoiceConfigError(f"whisper model.bin sha256 {whisper_sha[:12]} does not match "
                                       f"the configured {WHISPER_SHA256[:12]}")
            pipeline = Pipeline.from_pretrained(PIPELINE_PATH)
            pipeline.to(torch.device("cpu"))
            pipeline._segmentation.step = DIARIZATION_STEP * pipeline._segmentation.duration
            whisper = WhisperModel(WHISPER_PATH, device="cpu", compute_type=COMPUTE_TYPE,
                                   cpu_threads=THREADS, num_workers=1)
            voices = [VoiceModel(e, THREADS) for e in voice_entries]
        except VoiceConfigError as e:  # wrong files on the volume: do not retry into a loop
            model_error = str(e)
            log.error("%s", model_error)
            return
        except Exception as e:  # keep serving /health with the reason
            model_error = f"model load failed: {type(e).__name__}"
            log.exception("model load failed; retrying in 60 s")
            time.sleep(60)
            continue
        models.update(pipeline=pipeline, whisper=whisper, voices=voices)
        model_info.update(
            transcription={"model": WHISPER_ID, "version": WHISPER_REVISION or None,
                           "sha256": whisper_sha, "compute_type": COMPUTE_TYPE},
            diarization={"model": PIPELINE_ID, "version": PIPELINE_REVISION or None,
                         "step": DIARIZATION_STEP},
            voice=[{"model": v.entry["model"], "version": v.entry["version"],
                    "sha256": v.entry["sha256"]} for v in voices])
        model_error = ""
        log.info("models loaded: threads=%d compute_type=%s beam=%d diarization_step=%s voice_models=%d",
                 THREADS, COMPUTE_TYPE, BEAM_SIZE, DIARIZATION_STEP, len(voices))
        return


@app.on_event("startup")
async def startup():
    threading.Thread(target=load_models, daemon=True).start()
    app.state.reaper = asyncio.create_task(reap_jobs())


def job_view(job: dict) -> dict:
    out = {k: job[k] for k in ("id", "status", "created", "started", "finished") if job.get(k)}
    if job["status"] == "failed":
        out["error"] = job["error"]
    if job["status"] == "done":
        out["result"] = job["result"]
    return out


class Gate:
    """Plain ASGI middleware: refuse before a byte of the body is read.

    Not @app.middleware("http"): that wrapper hides the client's disconnect
    from request.is_disconnected(), and a cancelled request would then hold
    every CPU until its job ended. Checked here, in order: the key (401); for
    an upload, a known Idempotency-Key header on the job path (answer with
    that job, no upload read), a Content-Length (411) within MAX_UPLOAD_BYTES
    (413), models loaded (503), a free job slot (429 busy, with Retry-After).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        token = headers.get("authorization", "").removeprefix("Bearer ").strip()
        err = None
        if not hmac.compare_digest(token.encode(), API_KEY.encode()):
            err = ApiError(401, "unauthorized", "invalid api key")
        elif scope["method"] == "POST" and scope["path"].startswith("/v1/audio/transcriptions"):
            key = headers.get("idempotency-key", "")
            if scope["path"].endswith("/jobs") and key in idempotency and idempotency[key] in async_jobs:
                job = async_jobs[idempotency[key]]
                return await JSONResponse(job_view(job), status_code=200)(scope, receive, send)
            length = headers.get("content-length", "")
            if not length.isdigit():
                err = ApiError(411, "length_required", "Content-Length required")
            elif int(length) > MAX_UPLOAD_BYTES:
                err = ApiError(413, "too_long", f"upload larger than {MAX_UPLOAD_BYTES} bytes")
            elif model_error:
                err = ApiError(503, "internal", "models not loaded", {"Retry-After": "60"})
            elif jobs.locked():
                err = busy()
        if err:
            return await err.response()(scope, receive, send)
        return await self.app(scope, receive, send)


app.add_middleware(Gate)


def decode(fd: int, limit: float, path_name: str) -> np.ndarray:
    """Any audio ffmpeg reads -> 16 kHz mono float32, at most `limit` seconds.

    fd is the upload Starlette already spooled to disk. ffprobe and ffmpeg open
    it as /dev/fd/N (a fresh, seekable open of the same file), so the upload is
    never copied a second time. ffmpeg's own messages are not passed on: they
    can quote the file's metadata.
    """
    path = f"/dev/fd/{fd}"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "json", path],
        capture_output=True, check=False, pass_fds=(fd,))
    try:
        names = set(json.loads(probe.stdout)["format"]["format_name"].split(","))
    except (ValueError, KeyError):
        raise ApiError(400, "audio_unreadable", "not an audio file ffmpeg can read")
    if not names & AUDIO_FORMATS:
        raise ApiError(400, "audio_unreadable", f"unsupported container {','.join(sorted(names))}")
    # Decoded to a file, then read once: through a pipe the samples would sit in
    # memory twice (the bytes, then a writable copy for torch). 2 h = 460 MB.
    with tempfile.NamedTemporaryFile(suffix=".f32") as raw:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-t", str(limit + 1),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", raw.name],
            capture_output=True, check=False, pass_fds=(fd,))
        if proc.returncode != 0:
            raise ApiError(400, "audio_unreadable", "ffmpeg could not decode the audio")
        audio = np.fromfile(raw.name, dtype=np.float32)
    if not len(audio):
        raise ApiError(400, "audio_unreadable", "the file holds no audio")
    if len(audio) / SAMPLE_RATE > limit:
        hint = "; use POST /v1/audio/transcriptions/jobs" if path_name == "direct" and limit < MAX_SECONDS else ""
        raise ApiError(413, "too_long", f"audio longer than {limit:.0f} s for the {path_name} path{hint}")
    return audio


def diarize(audio: np.ndarray, cancel: threading.Event, **speakers):
    """-> (exclusive turns for words, overlap-aware turns for clean speech)."""
    import torch

    def hook(*args, **kwargs):  # pyannote calls this between and within steps
        if cancel.is_set():
            raise Cancelled()

    wav = torch.from_numpy(audio).unsqueeze(0)
    out = models["pipeline"]({"waveform": wav, "sample_rate": SAMPLE_RATE}, hook=hook, **speakers)
    # community-1 also returns an overlap-free version: every instant has at
    # most one speaker, which is what assigning each word to one speaker needs.
    full = getattr(out, "speaker_diarization", out)
    exclusive = getattr(out, "exclusive_speaker_diarization", None) or full
    return ([(t.start, t.end, s) for t, _, s in exclusive.itertracks(yield_label=True)],
            [(t.start, t.end, s) for t, _, s in full.itertracks(yield_label=True)])


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


def signatures(audio, full_turns, names, segments, cancel) -> list[dict]:
    """One `speakers` entry per label used in `segments`, in label order."""
    clean = clean_regions(full_turns)
    label_of = {v: k for k, v in names.items()}
    counts: dict = {}
    for seg in segments:
        counts[seg["speaker"]] = counts.get(seg["speaker"], 0) + 1
    out = []
    for label in sorted(counts, key=lambda s: int(s.rsplit(" ", 1)[-1])):
        regions = clean.get(label_of[label], [])
        entry = {"label": label, "speech_seconds": 0.0, "segment_count": counts[label], "embeddings": []}
        available = sum(e - s for s, e in regions)
        if available >= VOICE_MIN_SECONDS:
            for vm in models["voices"]:
                if cancel.is_set():
                    raise Cancelled()
                vec, used = vm.signature(audio, regions, SAMPLE_RATE)
                if vec is not None and used >= VOICE_MIN_SECONDS:
                    entry["embeddings"].append({"model": vm.entry["model"], "version": vm.entry["version"],
                                                "vector": [round(x, 6) for x in vec]})
                    entry["speech_seconds"] = round(max(entry["speech_seconds"], used), 1)
        if not entry["embeddings"]:
            entry["speech_seconds"] = round(available, 1)
        out.append(entry)
    return out


def run_job(audio, cancel, language, prompt, speakers, want_voice, job_id="direct", on_start=None) -> dict:
    """The CPU part of a job. Returns everything the formatter needs."""
    duration = len(audio) / SAMPLE_RATE
    with cpu_lock:
        current.update(started=time.monotonic(), audio_seconds=duration)
        if on_start:
            on_start()
        try:
            t0 = time.monotonic()
            turns, full_turns = diarize(audio, cancel, **speakers)
            t1 = time.monotonic()
            words, detected = transcribe_words(audio, cancel, language, prompt)
            t2 = time.monotonic()
            names: dict = {}
            segments = assign_speakers(words, turns, names)
            if not segments:
                raise ApiError(422, "no_speech", "no speech was found in the audio")
            voice = signatures(audio, full_turns, names, segments, cancel) if want_voice else None
            t3 = time.monotonic()
        finally:
            current.clear()
    log.info("job %s done: audio %.0f s, diarize %.0f s, transcribe %.0f s, signatures %.0f s, "
             "%d turns, %d words, %d speakers", job_id, duration, t1 - t0, t2 - t1, t3 - t2,
             len(turns), len(words), len({s["speaker"] for s in segments}))
    return {"segments": segments, "duration": duration, "language": language or detected, "speakers": voice}


def render(result: dict, response_format: str):
    segments, duration = result["segments"], result["duration"]
    labelled = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)
    usage = {"type": "duration", "seconds": int(round(duration))}
    if response_format == "text":
        return labelled
    if response_format == "verbose_json":
        body = {"task": "transcribe", "language": result["language"], "duration": round(duration, 2),
                "text": labelled, "segments": segments, "usage": usage}
    elif response_format == "diarized_json":
        body = {"task": "transcribe", "language": result["language"], "duration": round(duration, 2),
                "text": labelled,
                "segments": [{"type": "transcript.text.segment", **seg, "id": f"seg_{seg['id']}"}
                             for seg in segments],
                "usage": usage}
    else:
        body = {"text": labelled, "language": result["language"], "duration": round(duration, 2), "usage": usage}
    if result.get("speakers") is not None:
        body["speakers"] = result["speakers"]
    return body


def check_request(response_format, want_voice):
    """Re-checked here because Gate's checks and the job slot are not atomic."""
    if response_format not in FORMATS:
        raise ApiError(400, "invalid_request", f"unsupported response_format {response_format!r}")
    if want_voice and response_format == "text":
        raise ApiError(400, "invalid_request", "include_speaker_embeddings needs a JSON response_format")
    if model_error:
        raise ApiError(503, "internal", "models not loaded", {"Retry-After": "60"})
    if jobs.locked():
        raise busy()


async def receive(file: UploadFile, limit: float, path_name: str) -> np.ndarray:
    """Decode the spooled upload, then delete the spool (it would sit on disk for the whole job)."""
    try:
        spool = file.file
        if hasattr(spool, "rollover"):
            spool.rollover()  # a small upload may still be in memory; give it a file
        spool.flush()
        size = os.fstat(spool.fileno()).st_size
        if size > MAX_UPLOAD_BYTES:
            raise ApiError(413, "too_long", f"upload larger than {MAX_UPLOAD_BYTES} bytes")
        return await asyncio.to_thread(decode, spool.fileno(), limit, path_name)
    finally:
        await file.close()


def speaker_args(num_speakers, min_speakers, max_speakers):
    return {k: v for k, v in (("num_speakers", num_speakers), ("min_speakers", min_speakers),
                              ("max_speakers", max_speakers)) if v}


@app.get("/health")
async def health():
    if model_error:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return {"status": "ok", "busy": jobs.locked()}


@app.get("/v1/status")
async def status():
    return {
        "ready": not model_error,
        "detail": model_error or None,
        "busy": jobs.locked(),
        "retry_after": retry_after() if jobs.locked() else None,
        "models": model_info or None,
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "direct": {"max_audio_seconds": DIRECT_MAX_SECONDS},
            "jobs": {"max_audio_seconds": MAX_SECONDS, "result_retention_seconds": JOB_TTL},
            "concurrent_jobs": MAX_JOBS,
            "voice_min_clean_speech_seconds": VOICE_MIN_SECONDS,
        },
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),               # accepted, ignored: one model is served
    chunking_strategy: str | None = Form(None),   # accepted, ignored: chunking is internal
    response_format: str = Form("json"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    include_speaker_embeddings: bool = Form(False),
):
    check_request(response_format, include_speaker_embeddings)
    async with jobs:
        audio = await receive(file, DIRECT_MAX_SECONDS, "direct")
        cancel = threading.Event()

        async def watch():
            while not cancel.is_set():
                if await request.is_disconnected():
                    log.info("direct request: client went away; cancelling")
                    cancel.set()
                await asyncio.sleep(2)

        watcher = asyncio.create_task(watch())
        try:
            result = await asyncio.to_thread(run_job, audio, cancel, language, prompt,
                                             speaker_args(num_speakers, min_speakers, max_speakers),
                                             include_speaker_embeddings)
        except Cancelled:
            raise ApiError(499, "cancelled", "client went away")
        finally:
            cancel.set()
            watcher.cancel()
    body = render(result, response_format)
    return PlainTextResponse(body) if response_format == "text" else JSONResponse(body)


@app.post("/v1/audio/transcriptions/jobs", status_code=202)
async def submit_job(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(None),
    chunking_strategy: str | None = Form(None),
    response_format: str = Form("json"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    include_speaker_embeddings: bool = Form(False),
    idempotency_key: str | None = Form(None),
):
    key = request.headers.get("idempotency-key") or idempotency_key or None
    if key is not None and not IDEMPOTENCY_KEY.match(key):
        raise ApiError(400, "invalid_request", "idempotency key: 1-128 of A-Z a-z 0-9 . _ : -")
    if key and key in idempotency and idempotency[key] in async_jobs:
        await file.close()
        return JSONResponse(job_view(async_jobs[idempotency[key]]), status_code=200)
    check_request(response_format, include_speaker_embeddings)
    await jobs.acquire()
    try:
        audio = await receive(file, MAX_SECONDS, "job")
    except BaseException:
        jobs.release()
        raise
    job_id = uuid.uuid4().hex
    job = async_jobs[job_id] = {"id": job_id, "status": "queued", "created": time.time()}
    if key:
        idempotency[key] = job_id
    log.info("job %s queued: audio %.0f s", job_id, len(audio) / SAMPLE_RATE)

    def started():
        job.update(status="running", started=time.time())

    async def work():
        try:
            result = await asyncio.to_thread(run_job, audio, threading.Event(), language, prompt,
                                             speaker_args(num_speakers, min_speakers, max_speakers),
                                             include_speaker_embeddings, job_id, started)
            job.update(status="done", result=render(result, response_format))
        except ApiError as e:
            log.info("job %s failed: %s", job_id, e.reason)
            job.update(status="failed", error=e.body()["error"])
        except Exception as e:
            log.error("job %s failed: internal %s", job_id, type(e).__name__)
            job.update(status="failed", error={"reason": "internal", "message": "internal error"})
        finally:
            job["finished"] = time.time()
            jobs.release()

    job["_task"] = asyncio.create_task(work())  # keep a reference: the loop holds tasks weakly
    return JSONResponse(job_view(job), status_code=202)


@app.get("/v1/audio/transcriptions/jobs/{job_id}")
async def get_job(job_id: str):
    job = async_jobs.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", "no such job: finished jobs are purged after "
                       f"{JOB_TTL:.0f} s and a restart forgets all; resubmit with the same idempotency key")
    return job_view(job)


def purge(now: float):
    for job_id in [k for k, j in async_jobs.items() if j.get("finished") and now - j["finished"] > JOB_TTL]:
        async_jobs.pop(job_id, None)
        for key in [k for k, v in idempotency.items() if v == job_id]:
            idempotency.pop(key, None)
        log.info("job %s purged", job_id)


async def reap_jobs():
    while True:
        purge(time.time())
        await asyncio.sleep(60)
