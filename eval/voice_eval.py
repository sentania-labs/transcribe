"""Measure voice signatures against ground truth (not used by the service).

    python eval/voice_eval.py OUT.json NAME AUDIO OYEZ.json [START END] ...

For each recording (or part of one, START..END seconds): run the service's
diarization (same pipeline and step), map every diarized speaker to the real
person who talks most inside its turns (Oyez speaker timings), and save:
  - the pipeline's own per-speaker centroid (method A),
  - the clean-speech signature the service ships (method B, clean-mean-v1),
  - method B again from only N seconds of clean speech, for the threshold curve.
Analysis (cosine similarities) is eval/voice_report.py.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from pyannote.audio import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.assign import cpu_limit  # noqa: E402
from app.voice import VoiceModel, clean_regions, unit  # noqa: E402

RATE = 16000
PIPE = os.environ.get("DIARIZATION_PIPELINE", "/work/models/pyannote/speaker-diarization-community-1")
STEP = float(os.environ.get("DIARIZATION_STEP", 0.25))
BUDGETS = (5, 10, 20, 30, 60, 120)
ENTRY = {"model": "eval", "revision": "eval", "method": "clean-mean-v1",
         "path": os.path.join(PIPE, "embedding"),
         "sha256": "6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929"}


def load(path, start, end):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start), "-t", str(end - start)]
    cmd += ["-i", path, "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]
    return np.frombuffer(bytearray(subprocess.run(cmd, capture_output=True, check=True).stdout), dtype=np.float32)


def truth(oyez, start, end):
    d = json.load(open(oyez))
    out = []
    for sec in d["transcript"]["sections"]:
        for t in sec["turns"]:
            if not t.get("speaker"):
                continue
            a, b = t["start"], t["stop"]
            if start is not None:
                a, b = max(a, start) - start, min(b, end) - start
            if b > a:
                out.append((a, b, t["speaker"]["name"]))
    return out


def overlap(regions, gt):
    tally = {}
    for s, e in regions:
        for a, b, name in gt:
            ov = min(e, b) - max(s, a)
            if ov > 0:
                tally[name] = tally.get(name, 0.0) + ov
    return tally


def main():
    out_path, rest = sys.argv[1], sys.argv[2:]
    threads = cpu_limit()
    torch.set_num_threads(threads)
    pipe = Pipeline.from_pretrained(PIPE)
    pipe.to(torch.device("cpu"))
    pipe._segmentation.step = STEP * pipe._segmentation.duration
    vm = VoiceModel(ENTRY, threads)
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    i = 0
    while i < len(rest):
        name, audio_path, oyez = rest[i:i + 3]
        i += 3
        start = end = None
        if i + 1 < len(rest) and rest[i].replace(".", "").isdigit():
            start, end = float(rest[i]), float(rest[i + 1])
            i += 2
        t0 = time.monotonic()
        audio = load(audio_path, start, end)
        out = pipe({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": RATE})
        t1 = time.monotonic()
        turns = [(t.start, t.end, spk) for t, _, spk in out.speaker_diarization.itertracks(yield_label=True)]
        clean = clean_regions(turns)
        gt = truth(oyez, start, end)
        labels = out.speaker_diarization.labels()
        speakers = []
        for k, label in enumerate(labels):
            regions = clean.get(label, [])
            all_regions = [(s, e) for s, e, spk in turns if spk == label]
            tally = overlap(all_regions, gt)
            person = max(tally, key=tally.get) if tally else None
            total = sum(e - s for s, e in all_regions)
            entry = {"label": label, "person": person,
                     "purity": round(tally.get(person, 0.0) / total, 3) if total else 0.0,
                     "talk_seconds": round(total, 1),
                     "clean_seconds": round(sum(e - s for s, e in regions), 1)}
            cen = out.speaker_embeddings[k] if out.speaker_embeddings is not None else None
            entry["A_centroid"] = unit([float(x) for x in cen]) if cen is not None else None
            vec, used = vm.signature(audio, regions, RATE)
            entry["B_full"], entry["B_full_seconds"] = vec, round(used, 1)
            entry["B_budget"] = {}
            for n in BUDGETS:
                v, u = vm.signature(audio, regions, RATE, budget=n)
                entry["B_budget"][str(n)] = {"vector": v, "seconds": round(u, 1)}
            speakers.append(entry)
        t2 = time.monotonic()
        results[name] = {"audio_seconds": round(len(audio) / RATE, 1), "diarize_seconds": round(t1 - t0),
                         "signature_seconds": round(t2 - t1), "speakers": speakers}
        json.dump(results, open(out_path, "w"))
        print(f"{name}: {len(audio) / RATE:.0f} s audio, diarize {t1 - t0:.0f} s, signatures {t2 - t1:.0f} s, "
              f"{len(labels)} speakers", flush=True)


if __name__ == "__main__":
    main()
