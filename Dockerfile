# CPU-only speech-to-text with speaker labels. No model weights in the image:
# pyannote speaker-diarization-community-1 is licence-gated on Hugging Face and
# must never be published here. Both models are read from /models at runtime
# (see README.md).
FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt constraints.txt /tmp/
# CPU torch from the PyTorch index: the default PyPI wheel drags in CUDA.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      -c /tmp/constraints.txt -r /tmp/requirements.txt \
 && pip check \
 && python -c "import torch, torchaudio, pyannote.audio, faster_whisper; print('ok', torch.__version__, pyannote.audio.__version__, faster_whisper.__version__)"
RUN useradd --uid 10001 --create-home --home-dir /home/app app
COPY app /opt/transcribe/app
WORKDIR /opt/transcribe
USER 10001
ENV HF_HUB_OFFLINE=1 \
    PYANNOTE_METRICS_ENABLED=false \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
