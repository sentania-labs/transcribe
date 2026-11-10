"""Pure helpers: no model imports, so the tests run without torch."""
import bisect
import os

PAUSE_SPLIT = 5.0    # a pause this long starts a new segment, even for one speaker
MAX_SEGMENT = 60.0   # never let one segment grow past this (seconds)
SENTENCE_GAP = 1.0   # a pause this long also ends a sentence
SENTENCE_MAX = 40    # words; a sentence with no punctuation is cut here
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


def overlaps(start: float, end: float, turns, starts) -> dict:
    """{speaker: seconds of [start, end] inside that speaker's turns}.

    With no overlap at all, the nearest turn's speaker gets a token weight, so
    a word in a diarization gap still belongs to someone.
    """
    if not turns:
        return {}
    i = bisect.bisect_right(starts, end)
    out = {}
    # Turns are sorted and non-overlapping (exclusive diarization), so only the
    # turns starting before `end` and ending after `start` can overlap.
    j = i - 1
    while j >= 0 and turns[j][1] > start - 30:
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


def speaker_at(start: float, end: float, turns, starts) -> str | None:
    ov = overlaps(start, end, turns, starts)
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
    names, segments, cur = {}, [], None
    for sent in sentences(words):
        votes = {}
        for start, end, _ in sent:
            for spk, ov in overlaps(start, end, turns, starts).items():
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
