"""Cosine similarity report for eval/voice_eval.py output (no torch needed).

    python eval/voice_report.py OUT.json [OUT2.json ...]

Only speakers whose diarized turns are at least PURITY one real person, with
the person known, are used. Pairs are always across two different recordings
(or two non-overlapping parts of one).
"""
import itertools
import json
import math
import sys

PURITY = 0.8
MIN_CLEAN = 30.0
SESSION = {"A1": "2024-01-17", "A2": "2024-01-17", "B": "2024-01-17", "C": "2024-02-08", "D": "2024-03-18"}


def short(name):
    return name.replace(", Jr.", "").split()[-1]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else float("nan")


def summary(xs):
    return f"n={len(xs):4d} min={min(xs):.3f} p5={pct(xs, .05):.3f} mean={sum(xs) / len(xs):.3f} max={max(xs):.3f}"


def main():
    recs = {}
    for path in sys.argv[1:]:
        recs.update(json.load(open(path)))
    spk = []
    for rec, r in recs.items():
        for s in r["speakers"]:
            if s["person"] and s["purity"] >= PURITY:
                spk.append({**s, "rec": rec})
    print("recordings:", {k: (v["audio_seconds"], v["diarize_seconds"], v["signature_seconds"]) for k, v in recs.items()})
    print(f"speakers kept (purity >= {PURITY}): {len(spk)}; dropped: "
          f"{sum(len(r['speakers']) for r in recs.values()) - len(spk)}")
    for rec in recs:
        ppl = sorted((short(s["person"]), s["clean_seconds"]) for s in spk if s["rec"] == rec)
        print(f"  {rec}: " + ", ".join(f"{p} {c:.0f}s" for p, c in ppl))
    full = [s for s in spk if s["clean_seconds"] >= MIN_CLEAN]
    for method in ("A_centroid", "B_full"):
        same, diff, same_x = [], [], []
        for a, b in itertools.combinations(full, 2):
            if a["rec"] == b["rec"] or not a[method] or not b[method]:
                continue
            c = cos(a[method], b[method])
            if a["person"] == b["person"]:
                same.append(c)
                if SESSION[a["rec"]] != SESSION[b["rec"]]:
                    same_x.append(c)
            else:
                diff.append(c)
        print(f"\n{method} (speakers with >= {MIN_CLEAN:.0f} s clean speech, pairs across recordings)")
        if not same or not diff:
            print("  not enough cross-recording pairs")
            continue
        print(f"  same person          {summary(same)}")
        print(f"  same person, other day {summary(same_x) if same_x else 'n=0'}")
        print(f"  different people     {summary(diff)}")
        print("  threshold  false-reject(same)  false-accept(diff)")
        for t in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            fr = sum(c < t for c in same) / len(same)
            fa = sum(c >= t for c in diff) / len(diff)
            print(f"  {t:.2f}       {fr:6.1%}              {fa:6.1%}")
        worst = sorted(((cos(a[method], b[method]), a["person"], a["rec"], b["person"], b["rec"])
                        for a, b in itertools.combinations(full, 2)
                        if a["rec"] != b["rec"] and a["person"] != b["person"]), reverse=True)[:5]
        print("  closest different-person pairs:", [(round(c, 3), short(p), r, short(q), s)
                                                    for c, p, r, q, s in worst])
        low = sorted((cos(a[method], b[method]), a["person"], a["rec"], b["rec"])
                     for a, b in itertools.combinations(full, 2)
                     if a["rec"] != b["rec"] and a["person"] == b["person"])[:5]
        print("  lowest same-person pairs:", [(round(c, 3), short(p), r, s) for c, p, r, s in low])
    print("\nShort signatures: N seconds of clean speech vs the other recordings' full signatures (B)")
    print("  N    same: min   p5    mean | different: p95   max  | false-accept at 0.5 / false-reject at 0.5")
    for n in ("5", "10", "20", "30", "60", "120"):
        same, diff = [], []
        for a in spk:
            sv = a["B_budget"][n]["vector"]
            if not sv or a["B_budget"][n]["seconds"] < float(n) * 0.6:
                continue
            for b in full:
                if b["rec"] == a["rec"] or not b["B_full"]:
                    continue
                (same if b["person"] == a["person"] else diff).append(cos(sv, b["B_full"]))
        if same and diff:
            print(f"  {n:>3}  {min(same):.3f} {pct(same, .05):.3f} {sum(same) / len(same):.3f} | "
                  f"{pct(diff, .95):.3f} {max(diff):.3f} | {sum(c >= .5 for c in diff) / len(diff):.1%} / "
                  f"{sum(c < .5 for c in same) / len(same):.1%}")


if __name__ == "__main__":
    main()
