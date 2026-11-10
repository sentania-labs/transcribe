# transcribe

CPU-only speech-to-text with speaker labels behind an OpenAI-compatible
`POST /v1/audio/transcriptions`. Built for batch work on meeting recordings
(one to two hours) in the Sentania lab Kubernetes cluster; speed is not the
point, no GPU is needed.

- **Speaker turns**: pyannote `speaker-diarization-community-1` (pyannote.audio 4.0.7, CPU torch).
- **Text**: faster-whisper (CTranslate2) `large-v3-turbo`, int8, word timestamps, VAD filter.
- **Join**: each word goes to the speaker whose turn overlaps it most; runs of one
  speaker become segments (`app/assign.py`).

Threads follow the container's CPU limit (cgroup `cpu.max`), or `THREADS`.

## No weights in this repo or image

pyannote community-1 is licence-gated on Hugging Face. The image carries no
model files and never downloads any (`HF_HUB_OFFLINE=1`). Mount a volume at
`/models` laid out as:

```
/models/pyannote/speaker-diarization-community-1/   config.yaml, segmentation/, embedding/, plda/
/models/whisper/large-v3-turbo/                     model.bin, config.json, tokenizer.json, ...
```

The Whisper files are public (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`).
Until both are present the server runs, `/health` answers 503 naming the
missing files, and it retries every 30 s.

## API

```sh
curl -H "Authorization: Bearer $KEY" -F file=@meeting.m4a -F response_format=verbose_json \
  https://<host>/v1/audio/transcriptions
```

Form fields: `file` (required), `response_format` (`json` default, `text`,
`verbose_json`, `diarized_json`), `language`, `prompt`, `num_speakers`,
`min_speakers`, `max_speakers`. `model` is accepted and ignored.

Long recordings: `POST /v1/audio/transcriptions/jobs` (same form) answers `202
{"id","status"}`; poll `GET /v1/audio/transcriptions/jobs/{id}` until `status`
is `completed` (result under `result`, or `text` for `response_format=text`) or
`failed`. Jobs are in memory: kept `JOB_TTL` seconds (default 24 h) after they
finish and lost on restart.

## Limits and settings (environment)

| var | default | |
|---|---|---|
| `TRANSCRIBE_API_KEY` | required | bearer key, checked before the upload is read; uploads need a Content-Length (411 otherwise) |
| `MAX_UPLOAD_BYTES` | 1 GiB | larger uploads get 413 |
| `MAX_SECONDS` | 9000 | longer audio gets 413 |
| `MAX_JOBS` | 1 | more at once get 429, before the upload is read |
| `WHISPER_WINDOW` | 1200 | seconds of audio per Whisper pass, cut at silence; bounds feature memory (about 3 GB per hour of input otherwise) |
| `DIARIZATION_STEP` | 0.25 | pyannote sliding-window step as a fraction of its 10 s window (pyannote default 0.1: 2.6x slower, same turns on the test clip; 0.5 fragments turns) |
| `WHISPER_BEAM_SIZE` | 5 | 1 is greedy and faster |
| `WHISPER_COMPUTE_TYPE` | int8 | CTranslate2 compute type |
| `JOB_TTL` | 86400 | seconds a finished async job is kept |

Only ffmpeg containers in an allowlist are decoded (no playlists or concat
lists). A synchronous request whose client disconnects is cancelled.

## Develop

```sh
scripts/check.sh                        # syntax + unit tests, no torch needed
docker build -t transcribe:dev .
scripts/smoke.sh transcribe:dev         # boots without models: 503 health, 401/503 API
```

Release: tag `vX.Y.Z` on `main`; `release.yml` builds, smokes and pushes
`ghcr.io/sentania-labs/transcribe:vX.Y.Z`. Deployments pin the digest.
`constraints.txt` is a `pip freeze` of a verified build; regenerate it with
`docker run --rm --entrypoint pip transcribe:dev freeze`.
