"""Voice signatures: one unit-length vector per speaker, per configured voice model.

The service computes and returns them; it never stores or matches them (the
caller owns that). A signature is built only from a speaker's CLEAN speech:
time when that speaker, and nobody else, is talking according to the
overlap-aware diarization. Below VOICE_MIN_SECONDS of clean speech the entry
is left out: no signature is better than a bad one.

Voice models come from configuration (VOICE_MODELS, JSON list), never from
the image, so an image update cannot change what a vector means. Each entry:

  model     name reported on every vector
  revision  the pinned upstream revision of the weights
  method    how a speaker's vector is built (METHODS below)
  path      directory of the embedding model (pytorch_model.bin inside)
  sha256    of pytorch_model.bin; the service refuses the entry on a mismatch

The reported version is "<revision>+<method>+<fingerprint>". The fingerprint
is a hash, computed at start, of everything that shapes the vector: the
weights, and for pipeline-centroid-v1 also the pipeline's config.yaml,
segmentation weights, plda files and the diarization step. So new weights or a
re-tuned pipeline always give a new version, even if nobody edits `revision`.
Two entries during a changeover give two vectors per speaker.
"""
import hashlib
import json
import math
import os

# Clean speech per embedding window: pieces longer than this are split, and
# shorter pieces are dropped (the model is unreliable on very short audio).
WINDOW_MAX = 10.0
WINDOW_MIN = 2.0
# At most this much clean speech per speaker goes into a signature, spread
# evenly over the recording: bounds the CPU cost for a two-hour monologue.
SPEAKER_BUDGET = 300.0
# pipeline-centroid-v1: the diarization pipeline's own per-speaker centroid
#   (pyannote community-1: VBx-weighted mean of its 10 s chunk embeddings,
#   overlap excluded). Only for the embedding model the pipeline itself uses.
#   Measured best (docs/voice-eval.md): same person across recordings >= 0.795,
#   different people <= 0.447.
# clean-mean-v1: this service's own windows over clean speech, embedded with
#   any configured model. Measured weaker (different people up to 0.781); kept
#   because it is the only way to run a SECOND model during a changeover.
METHODS = ("pipeline-centroid-v1", "clean-mean-v1")


class VoiceConfigError(ValueError):
    pass


def parse_voice_models(raw: str | None) -> list[dict]:
    """VOICE_MODELS -> validated entries. Empty or unset means no voice models."""
    if not raw or not raw.strip():
        return []
    try:
        entries = json.loads(raw)
    except ValueError as e:
        raise VoiceConfigError(f"VOICE_MODELS is not JSON: {e}") from None
    if not isinstance(entries, list):
        raise VoiceConfigError("VOICE_MODELS must be a JSON list")
    out, seen = [], set()
    for e in entries:
        if not isinstance(e, dict):
            raise VoiceConfigError("each VOICE_MODELS entry must be an object")
        missing = [k for k in ("model", "revision", "method", "path", "sha256") if not e.get(k)]
        if missing:
            raise VoiceConfigError(f"VOICE_MODELS entry missing {', '.join(missing)}")
        if e["method"] not in METHODS:
            raise VoiceConfigError(f"unknown voice method {e['method']!r} (this image knows {', '.join(METHODS)})")
        key = (e["model"], e["revision"], e["method"], e["sha256"])
        if key in seen:
            raise VoiceConfigError(f"VOICE_MODELS lists {e['model']} {e['revision']} {e['method']} twice")
        seen.add(key)
        # The final version gets the fingerprint at load time (VoiceModel).
        out.append({**e, "version": f"{e['revision']}+{e['method']}"})
    return out


def fingerprint(entry: dict, pipeline_dir: str | None, step: float) -> str:
    """12 hex characters over every input that shapes this entry's vectors."""
    h = hashlib.sha256()
    h.update(f"{entry['method']}|{entry['sha256']}".encode())
    if entry["method"] == "pipeline-centroid-v1":
        files = ["config.yaml", "segmentation/pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz"]
        for f in files:
            p = os.path.join(pipeline_dir, f)
            h.update(f"|{f}:{file_sha256(p) if os.path.exists(p) else 'missing'}".encode())
        h.update(f"|step:{step}".encode())
    return h.hexdigest()[:12]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def clean_regions(turns) -> dict:
    """Overlap-aware turns [(start, end, speaker)] -> {speaker: [(start, end)]} of
    the time that speaker is the ONLY one talking."""
    events = []
    for s, e, spk in turns:
        if e > s:
            events.append((s, 1, spk))
            events.append((e, -1, spk))
    events.sort(key=lambda x: (x[0], x[1]))  # ends before starts at the same instant
    active: dict = {}
    out: dict = {}
    last = None
    for t, kind, spk in events:
        if last is not None and t > last and len(active) == 1:
            (only,) = active
            regions = out.setdefault(only, [])
            if regions and abs(regions[-1][1] - last) < 1e-9:
                regions[-1] = (regions[-1][0], t)
            else:
                regions.append((last, t))
        if kind == 1:
            active[spk] = active.get(spk, 0) + 1
        else:
            active[spk] -= 1
            if not active[spk]:
                del active[spk]
        last = t
    return out


def embedding_windows(regions, budget: float = SPEAKER_BUDGET):
    """Clean regions -> [(start, end)] windows of WINDOW_MIN..WINDOW_MAX seconds,
    at most `budget` seconds in all, evenly spread when there is more."""
    pieces = []
    for s, e in regions:
        n = max(1, math.ceil((e - s) / WINDOW_MAX))
        step = (e - s) / n
        for k in range(n):
            a, b = s + k * step, s + (k + 1) * step
            if b - a >= WINDOW_MIN:
                pieces.append((a, b))
    total = sum(b - a for a, b in pieces)
    if total <= budget:
        return pieces
    # Evenly spaced pieces across the whole recording, so the signature is not
    # all from the first minutes.
    k = max(1, int(budget // (total / len(pieces))))
    if k == 1:
        idx = [len(pieces) // 2]
    else:
        idx = sorted({round(i * (len(pieces) - 1) / (k - 1)) for i in range(k)})
    keep = [pieces[i] for i in idx]
    while len(keep) > 1 and sum(b - a for a, b in keep) > budget:
        keep.pop(len(keep) // 2)
    return keep


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 and math.isfinite(n) else None


def mean_signature(vectors, weights):
    """Duration-weighted mean of unit vectors, made unit length again."""
    acc, total = None, 0.0
    for v, w in zip(vectors, weights):
        u = unit(v)
        if u is None:
            continue
        acc = [w * x for x in u] if acc is None else [a + w * x for a, x in zip(acc, u)]
        total += w
    return unit(acc) if acc else None


class VoiceModel:
    """A loaded voice model entry: embeds windows of 16 kHz mono float32 audio.

    pipeline_embedding: the directory of the embedding model the diarization
    pipeline uses; a pipeline-centroid-v1 entry must point at exactly it.
    """

    def __init__(self, entry: dict, threads: int, pipeline_dir: str | None = None, step: float = 0.0):
        weights = os.path.join(entry["path"], "pytorch_model.bin")
        if not os.path.exists(weights):
            raise VoiceConfigError(f"{entry['model']}: {weights} missing")
        digest = file_sha256(weights)
        if digest != entry["sha256"]:
            raise VoiceConfigError(f"{entry['model']}: pytorch_model.bin sha256 {digest[:12]} "
                                   f"does not match the configured {entry['sha256'][:12]}")
        self.entry = {**entry, "version": f"{entry['version']}+{fingerprint(entry, pipeline_dir, step)}"}
        self.embedding = None
        if entry["method"] == "pipeline-centroid-v1":
            own = os.path.join(pipeline_dir or "", "embedding")
            if not pipeline_dir or os.path.realpath(entry["path"]) != os.path.realpath(own):
                raise VoiceConfigError(f"{entry['model']}: pipeline-centroid-v1 needs the path of the "
                                       f"diarization pipeline's own embedding model ({own})")
            return
        try:
            import torch
            # The pyannote loader class directly: PretrainedSpeakerEmbedding()
            # picks a backend by substring of the path ("wespeaker", "nvidia",
            # ...), which could load something other than the file hashed above.
            from pyannote.audio.pipelines.speaker_verification import PyannoteAudioPretrainedSpeakerEmbedding
            torch.set_num_threads(threads)
            self.embedding = PyannoteAudioPretrainedSpeakerEmbedding(entry["path"], device=torch.device("cpu"))
        except Exception as e:
            raise VoiceConfigError(f"{entry['model']}: could not load ({type(e).__name__})") from None

    def embed(self, audio, windows, rate: int, batch: int = 16):
        """[(start, end)] seconds -> [vector or None], one per window."""
        import numpy as np
        import torch
        out = []
        for i in range(0, len(windows), batch):
            group = windows[i:i + batch]
            chunks = [audio[int(a * rate):int(b * rate)] for a, b in group]
            n = max(len(c) for c in chunks)
            wav = np.zeros((len(group), 1, n), dtype=np.float32)
            mask = np.zeros((len(group), n), dtype=np.float32)
            for j, chunk in enumerate(chunks):
                wav[j, 0, :len(chunk)] = chunk
                mask[j, :len(chunk)] = 1.0
            with torch.inference_mode():
                emb = self.embedding(torch.from_numpy(wav), masks=torch.from_numpy(mask))
            for v in emb:
                out.append(None if not np.all(np.isfinite(v)) else [float(x) for x in v])
        return out

    def signature(self, audio, regions, rate: int, budget: float = SPEAKER_BUDGET):
        """Clean regions of one speaker -> (vector or None, seconds used)."""
        wins = embedding_windows(regions, budget)
        if not wins:
            return None, 0.0
        vecs = self.embed(audio, wins, rate)
        used = [(v, b - a) for v, (a, b) in zip(vecs, wins) if v is not None]
        if not used:
            return None, 0.0
        return mean_signature([v for v, _ in used], [w for _, w in used]), sum(w for _, w in used)
