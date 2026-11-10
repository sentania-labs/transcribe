"""Pure helpers: no model imports, so the tests run without torch."""
import bisect
import os

PAUSE_SPLIT = 5.0    # a pause this long starts a new segment, even for one speaker
MAX_SEGMENT = 60.0   # never let one segment grow past this (seconds)
SENTENCE_GAP = 1.0   # a pause this long also ends a sentence
SENTENCE_MAX = 40    # words; a sentence with no punctuation is cut here
MIN_RUN = 8          # words; a shorter run of another speaker inside a sentence is timing noise
                     # (4 split real sentences on the test recording: diarization edges drift ~1 s)
SENTENCE_END = (".", "?", "!", "—", "…")  # dash: an interrupted sentence


def cpu_limit() -> int:
    """CPUs this container may use: the cgroup v2 quota, else the CPU count."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def windows(audio, rate: int, size: float, search: float = 30.0):
    """[(start, end)] seconds covering audio in pieces of at most `size`, each cut
    at the quietest 100 ms in the last `search` seconds before the limit."""
    total = len(audio) / rate
    start, out = 0.0, []
    frame = rate // 10
    while total - start > size:
        lo, hi = int((start + size - search) * rate), int((start + size) * rate)
        chunk = audio[lo:hi][: (hi - lo) // frame * frame].reshape(-1, frame)
        cut = (lo + int(((chunk.astype("float64") ** 2).mean(axis=1)).argmin()) * frame) / rate
        out.append((start, cut))
        start = cut
    out.append((start, total))
    return out


def overlaps(start: float, end: float, turns, starts, longest: float = 0.0) -> dict:
    """{speaker: seconds of [start, end] inside that speaker's turns}.

    With no overlap at all, the nearest turn's speaker gets a token weight, so
    a word in a diarization gap still belongs to someone.
    """
    if not turns:
        return {}
    i = bisect.bisect_right(starts, end)
    out = {}
    # Turns are sorted by start; one that overlaps [start, end] starts before
    # `end` and no earlier than `start - longest` (the longest turn's length).
    longest = longest or max(e - s for s, e, _ in turns)
    j = i - 1
    while j >= 0 and turns[j][0] >= start - longest:
        s, e, spk = turns[j]
        ov = min(e, end) - max(s, start)
        if ov > 0:
            out[spk] = out.get(spk, 0.0) + ov
        j -= 1
    if out:
        return out
    mid = (start + end) / 2
    near = [turns[k] for k in (i - 1, i) if 0 <= k < len(turns)]
    spk = min(near, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2]
    return {spk: 1e-6}


def speaker_at(start: float, end: float, turns, starts, longest: float = 0.0) -> str | None:
    ov = overlaps(start, end, turns, starts, longest)
    return max(ov, key=ov.get) if ov else None


def sentences(words):
    """Split [(start, end, text)] at sentence ends, long pauses and SENTENCE_MAX words."""
    cur = []
    for w in words:
        if cur and (w[0] - cur[-1][1] >= SENTENCE_GAP or len(cur) >= SENTENCE_MAX):
            yield cur
            cur = []
        cur.append(w)
        if w[2].strip().endswith(SENTENCE_END):
            yield cur
            cur = []
    if cur:
        yield cur


def split_runs(sent, turns, starts, longest):
    """A sentence -> pieces, split where the word-level speaker changes for at
    least MIN_RUN words. Whisper sometimes leaves a speaker change unpunctuated;
    shorter runs are timing noise at a turn edge and stay with their neighbours."""
    runs = []
    for w in sent:
        spk = speaker_at(w[0], w[1], turns, starts, longest)
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(w)
        else:
            runs.append([spk, [w]])
    while len(runs) > 1:
        short = [k for k, r in enumerate(runs) if len(r[1]) < MIN_RUN]
        if not short:
            break
        k = short[0]
        # Fold into the longer neighbour, then join neighbours left equal.
        n = k - 1 if k == len(runs) - 1 or (k > 0 and len(runs[k - 1][1]) >= len(runs[k + 1][1])) else k + 1
        if n < k:
            runs[n][1].extend(runs[k][1])
        else:
            runs[n][1][:0] = runs[k][1]
        del runs[k]
        merged = [runs[0]]
        for r in runs[1:]:
            if r[0] == merged[-1][0]:
                merged[-1][1].extend(r[1])
            else:
                merged.append(r)
        runs = merged
    return [r[1] for r in runs]


def assign_speakers(words, turns):
    """words [(start, end, text)], turns [(start, end, speaker)] -> segments.

    Each sentence goes to the speaker who covers most of its words' time.
    Word by word, the words at a turn change land on the wrong side whenever
    Whisper's and pyannote's timings disagree by a fraction of a second; whole
    sentences do not. A run of one speaker's sentences is a segment, split at a
    pause of PAUSE_SPLIT or when it would pass MAX_SEGMENT. Speakers are renamed
    "Speaker N" in order of first speech.
    """
    turns = sorted(turns)
    starts = [t[0] for t in turns]
    longest = max((e - s for s, e, _ in turns), default=0.0)
    pieces = [p for sent in sentences(words) for p in split_runs(sent, turns, starts, longest)]
    names, segments, cur = {}, [], None
    for sent in pieces:
        votes = {}
        for start, end, _ in sent:
            for spk, ov in overlaps(start, end, turns, starts, longest).items():
                votes[spk] = votes.get(spk, 0.0) + ov
        spk = max(votes, key=votes.get) if votes else "unknown"
        start, end = sent[0][0], sent[-1][1]
        text = [w[2] for w in sent]
        if (cur and cur["_spk"] == spk and start - cur["end"] < PAUSE_SPLIT
                and end - cur["start"] <= MAX_SEGMENT):
            cur["end"] = end
            cur["_text"].extend(text)
            continue
        names.setdefault(spk, f"Speaker {len(names) + 1}")
        cur = {"_spk": spk, "start": start, "end": end, "_text": text}
        segments.append(cur)
    out = []
    for seg in segments:
        text = "".join(seg["_text"]).strip()
        if not text:
            continue
        out.append({"id": len(out), "speaker": names[seg["_spk"]],
                    "start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": text})
    return out
