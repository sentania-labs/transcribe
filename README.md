# transcribe

CPU-only speech-to-text with speaker labels, and optional per-speaker voice
signatures, behind an OpenAI-compatible `POST /v1/audio/transcriptions`. Built
for batch work on meeting recordings (one to two hours) in the Sentania lab
Kubernetes cluster; speed is not the point, no GPU is needed.

- **Speaker turns**: pyannote `speaker-diarization-community-1` (pyannote.audio 4.0.7, CPU torch).
- **Text**: faster-whisper (CTranslate2) `large-v3-turbo`, int8, word timestamps, VAD filter,
  20-minute windows cut at silence.
- **Join**: each sentence goes to the speaker who covers most of its words' time; a run of 8+
  words from another speaker inside an unpunctuated sentence splits it (`app/assign.py`).
- **Voice signatures** (opt in): one unit-length vector per speaker and voice model; by
  default the diarization pipeline's own speaker centroid (`app/voice.py`). The service
  never stores or matches them.

Threads follow the container's CPU limit (cgroup `cpu.max`), or `THREADS`. About 0.41 x
the audio length on 4 CPUs (a 76 min recording took 31.5 min); signatures add a little.

## No weights in this repo or image

pyannote community-1 is licence-gated on Hugging Face. The image carries no model files
and never downloads any (`HF_HUB_OFFLINE=1`). Mount a volume at `/models`:

```
/models/pyannote/speaker-diarization-community-1/   config.yaml, segmentation/, embedding/, plda/
/models/whisper/large-v3-turbo/                     model.bin, config.json, tokenizer.json, ...
```

Whisper: `dropbox-dash/faster-whisper-large-v3-turbo` (formerly `mobiuslabsgmbh/...`,
public). Until the files are present the server runs, `/health` answers 503,
`/v1/status` names the missing files, and it retries every 30 s.

## API

Every call except `/health` needs `Authorization: Bearer <TRANSCRIBE_API_KEY>`.

### `POST /v1/audio/transcriptions` (direct)

```sh
curl -H "Authorization: Bearer $KEY" -F file=@meeting.m4a -F response_format=diarized_json \
  -F num_speakers=4 -F include_speaker_embeddings=true https://<host>/v1/audio/transcriptions
```

| form field | |
|---|---|
| `file` | required; any audio ffmpeg reads from an allowlist of containers (no playlists or concat lists) |
| `response_format` | `json` (default), `text`, `verbose_json`, `diarized_json` |
| `model`, `chunking_strategy` | accepted and ignored (one model is served; chunking is internal) |
| `language` | ISO code; detected from the first window when absent |
| `prompt` | Whisper initial prompt: names and terms to spell right; applies to the start of each 20-minute window |
| `num_speakers`, `min_speakers`, `max_speakers` | speaker-count hints for pyannote. For an attendee list prefer `max_speakers`: invitees who never speak make `num_speakers` an over-count, which adds stray labels (measured, `docs/voice-eval.md`) |
| `include_speaker_embeddings` | `true` adds `speakers` (below); JSON formats only |

Response (`diarized_json`; `verbose_json` is the same with plain segments, `{id, speaker,
start, end, text}`):

```json
{
  "task": "transcribe", "language": "en", "duration": 4572.65,
  "text": "Speaker 1: ...\nSpeaker 2: ...",
  "segments": [{"type": "transcript.text.segment", "id": "seg_0", "speaker": "Speaker 1",
                "start": 0.0, "end": 7.1, "text": "..."}],
  "usage": {"type": "duration", "seconds": 4573},
  "speakers": [
    {"label": "Speaker 1", "speech_seconds": 300.0, "segment_count": 14,
     "embeddings": [{"model": "pyannote/speaker-diarization-community-1/embedding",
                     "version": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee+pipeline-centroid-v1+<12 hex>",
                     "vector": [0.0123, -0.0456, "... 256 numbers"]}]}
  ]
}
```

`speakers` (only when asked): one entry per label used in `segments`, same label text.
`segment_count` counts that label's segments. `embeddings` holds one entry per configured
voice model, each a unit-length vector (compare with plain cosine, only within one
`model` + `version`). A speaker with less than `VOICE_MIN_SECONDS` of clean speech (time
when they and nobody else talk) gets no entry: `embeddings` may be empty.
`speech_seconds` is the speaker's clean speech in the recording (alone, no overlap): what
the signature is built from, and what the `VOICE_MIN_SECONDS` floor (default 20) is
checked against.

The text is `Speaker N: ...` lines; `text` format returns only that.

### Job path, for long recordings

`POST /v1/audio/transcriptions/jobs` takes the same form plus an optional idempotency key
(header `Idempotency-Key`, or form field `idempotency_key`; 1-128 of `A-Z a-z 0-9 . _ : -`).
It answers `202` with the job; a repeat with a known key answers `200` with the existing
job, whatever its state, until that job is purged. Exception: a job that failed with
`internal` (retryable) releases its key, so a resubmit with the same key runs again. With the header, a repeat is answered
before the upload is read, even while the service is busy; with only the form field, a
repeat sent while busy gets `429 busy` (the key is in the body, which is not read then).

`GET /v1/audio/transcriptions/jobs/{id}`:

```json
{"id": "...", "status": "queued|running|done|failed", "created": 1794273236.5,
 "started": ..., "finished": ...,
 "result": {...exactly the direct call's JSON...},
 "error": {"reason": "no_speech", "message": "..."}}
```

`result` only when `done` (a string for `response_format=text`), `error` only when
`failed`. A finished job is purged `JOB_TTL` seconds after it finishes (default 2 h);
then the id answers 404 `not_found` and the key is free again. **Jobs live in memory:
a restart loses them all.** The recovery is to resubmit with the same idempotency key.
Upload and decode errors on the job path are answered at submit time (no job is created).

### Errors

Every error is `{"error": {"reason": "<code>", "message": "<plain text>"}}`:

| reason | HTTP | meaning | retry? |
|---|---|---|---|
| `busy` | 429 + `Retry-After` (seconds, estimated from the running job) | a job is running | yes |
| `audio_unreadable` | 400 | not decodable audio, unsupported container, empty | no |
| `no_speech` | 422 | nothing transcribable | no |
| `too_long` | 413 | upload over `MAX_UPLOAD_BYTES`, or audio over this path's limit (the direct path's message points to the job path) | no, or use the job path |
| `internal` | 500, or 503 + `Retry-After` while models load | anything on the service side | yes, with backoff |
| `unauthorized` | 401 | bad or missing key | no |
| `invalid_request` | 400 | bad field or format | no |
| `length_required` | 411 | no Content-Length on an upload | no |
| `not_found` | 404 | unknown or purged job | resubmit |
| `cancelled` | 499 | the direct caller disconnected (nobody receives it) | n/a |

### `GET /v1/status`

Ready state, `busy` and `retry_after`, the loaded models (transcription
`model`/`version`/`sha256`, diarization, every voice model with `model`/`version`/`sha256`),
and the limits of both paths, job retention and `VOICE_MIN_SECONDS`.

`GET /health` (no auth) is the probe: 200 or 503.

## Voice models

Configured, never implied by the image. `VOICE_MODELS` is a JSON list; empty means no
signatures (`embeddings` is always empty). Each entry:

```json
{"model": "pyannote/speaker-diarization-community-1/embedding",
 "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
 "method": "pipeline-centroid-v1",
 "path": "/models/pyannote/speaker-diarization-community-1/embedding",
 "sha256": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929"}
```

`model` names where the weights come from (community-1's `embedding/`, itself a copy of
`pyannote/wespeaker-voxceleb-resnet34-LM`) and `revision` is that repository's commit. The
reported `version` is `<revision>+<method>+<fingerprint>`. The fingerprint (12 hex) is
computed at start over everything that shapes the vector: the weights, and for
`pipeline-centroid-v1` also the pipeline's `config.yaml` (clustering settings),
segmentation weights, `plda/` files and `DIARIZATION_STEP`. New weights or a re-tuned
pipeline therefore always give a new version, even if `revision` is not edited. A weights
file that does not match the configured `sha256` disables signatures (status
`voice_error`; requests with `include_speaker_embeddings=true` get `503 internal`) while
transcription keeps working. A changeover is a second entry:
both run and each speaker gets two vectors; removing the old entry ends it. Changing
`VOICE_MODELS` is the only way to change voice models: an image update keeps the vectors
identical as long as it supports the configured `method` (an image that drops a method
refuses to start with that config rather than guess).

Methods:

- `pipeline-centroid-v1` (deployed): the diarization pipeline's own per-speaker centroid,
  i.e. pyannote community-1's VBx-weighted mean of its 10 s chunk embeddings with
  overlapped speech excluded (a plain mean per cluster when a speaker count is forced),
  unit-normalised. Costs nothing extra. Only valid for the pipeline's own embedding model:
  the entry's `path` must be the pipeline's `embedding/` directory, or the service refuses.
- `clean-mean-v1`: the speaker's clean regions cut into 2-10 s windows, at most 300 s
  spread over the recording, each embedded, duration-weighted mean, unit-normalised. Works
  with any pyannote embedding model, so it is how a second model can run during a
  changeover. Measured clearly weaker on the same model (different people up to 0.78
  cosine against 0.45 for the centroid; `docs/voice-eval.md`).

## Privacy

No audio, transcript, prompt or vector is written to logs at any level (library loggers
are held at WARNING; every unhandled error is caught by the outermost middleware and logs
the path and exception type only, no traceback; ffmpeg's messages are not passed on
because they can quote file metadata; validation errors name fields, never values). Logs carry job ids, durations,
sizes, counts and reasons. Uploads are deleted once decoded. Nothing is kept after the
answer except a job's result until it is purged.

## Limits and settings (environment)

| var | default | |
|---|---|---|
| `TRANSCRIBE_API_KEY` | required | bearer key, checked before the upload is read |
| `MAX_UPLOAD_BYTES` | 1 GiB | larger uploads: `too_long` |
| `DIRECT_MAX_SECONDS` | 8400 | audio limit on the direct path (a synchronous call must end inside the ingress's 3600 s) |
| `MAX_SECONDS` | 9000 | audio limit on the job path |
| `JOB_TTL` | 7200 | seconds a finished job's result is kept |
| `VOICE_MODELS` | empty | voice models, above |
| `VOICE_MIN_SECONDS` | 20 | clean speech needed for a signature |
| `WHISPER_MODEL_ID`, `WHISPER_MODEL_REVISION`, `WHISPER_MODEL_SHA256` | `large-v3-turbo`, empty, empty | reported in status; a set sha256 is verified at start |
| `DIARIZATION_MODEL_ID`, `DIARIZATION_MODEL_REVISION` | community-1, empty | reported in status |
| `WHISPER_WINDOW` | 1200 | seconds of audio per Whisper pass, cut at silence; bounds feature memory |
| `DIARIZATION_STEP` | 0.25 | pyannote sliding-window step as a fraction of its 10 s window (0.1: 2.6x slower, same turns; 0.5 fragments turns) |
| `WHISPER_BEAM_SIZE` | 5 | 1 is greedy and faster |
| `WHISPER_COMPUTE_TYPE` | int8 | CTranslate2 compute type |
| `WORK_RATIO` | 0.45 | seconds of work per second of audio, for `Retry-After` |

## Evaluation

`eval/voice_eval.py` and `eval/voice_report.py` measure signatures against ground truth
(public-domain Supreme Court arguments with Oyez speaker timings); results and the
recommended match threshold are in `docs/voice-eval.md`.

## Develop

```sh
scripts/check.sh                        # syntax + unit tests, no torch needed
docker build -t transcribe:dev .
scripts/smoke.sh transcribe:dev         # boots without models: errors, status, auth
```

Release: tag `vX.Y.Z` on `main`; `release.yml` builds, smokes and pushes
`ghcr.io/sentania-labs/transcribe:vX.Y.Z`. Deployments pin the digest.
`constraints.txt` is a `pip freeze` of a verified build; regenerate it with
`docker run --rm --entrypoint pip transcribe:dev freeze`.
