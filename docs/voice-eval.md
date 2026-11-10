# Voice signature evaluation (2026-11-10)

Does the same person in two recordings give close vectors, and different people
not? Measured, with ground truth, before choosing a method and thresholds.

## Data

Public-domain U.S. Supreme Court oral arguments, with Oyez's speaker-timed
transcripts as ground truth (each diarized speaker is mapped to the real person
who talks most inside its turns; every label in every recording was at least
81.7% one person, and no person was split across two labels).

| id | case | date | audio | people |
|---|---|---|---|---|
| A1, A2 | 22-451 Loper Bright, first and second half (non-overlapping) | 2024-01-17 | 38 + 38 min | 9 + 9 |
| B | 22-1219 Relentless | 2024-01-17 | 132 min | 11 |
| C | 23-719 Trump v. Anderson | 2024-02-08 | 129 min | 12 |
| D | 22-842 NRA v. Vullo | 2024-03-18 | 74 min | 12 |

The nine justices recur across all of them, Prelogar is in A2 and B, Clement in
A1 and A2. 53 speaker instances. Pairs are always across two different
recordings (or halves); 74 same-person pairs (53 of them across different days)
and 958 different-person pairs, among speakers with at least 30 s of clean speech.

Run with the service's own pipeline settings (community-1, step 0.25) on 4 CPUs:
`eval/voice_eval.py` (vectors), `eval/voice_report.py` (numbers).

## Results: cosine similarity

| method | same person min / p5 / mean | same person, other day min | different people mean / p95 / max |
|---|---|---|---|
| **A: pipeline centroid** (`pipeline-centroid-v1`, deployed) | **0.795** / 0.871 / 0.929 | 0.795 | 0.114 / ~0.33 / **0.447** |
| B: mean over clean-speech windows (`clean-mean-v1`) | 0.797 / 0.835 / 0.913 | 0.797 | 0.287 / ~0.60 / 0.781 |

Error rates at a match threshold, method A: no false accept and no false reject
anywhere from 0.45 to 0.79. At 0.40, 1.0% false accepts; at 0.35, 4.7%.
Method B overlaps: 7.1% false accepts at 0.50, 0.5% even at 0.70.

Closest different people (A): Kagan vs Sotomayor, 0.447, 0.440, 0.419, 0.414.
Lowest same person (A): Gorsuch A1 vs C, 0.795 (Gorsuch has 33 s of clean speech in A1).

So A separates cleanly on this data, B does not. Same-day vs other-day made no
real difference for A (same-person minimum 0.795 either way).

## Little speech

The centroid is built from pyannote's 10 s chunks, so short speakers still get
a usable vector: Thomas with 23 s of clean speech scored 0.913 to 0.933 against
his own long references, and with 7 s, 0.900 to 0.907; different people stayed
under 0.26 for both. That is two speakers, not a curve.

Method B gives the curve (N seconds of clean speech against other recordings'
full signatures): at 5 s the same-person minimum is 0.35 and 18% fall under
0.5; at 10 s, 0.35 / 2.8%; from 20 s, no false rejects at 0.5.

## Forced speaker count

Coppermind will send the attendee count. With the correct count (A1 and A2,
`num_speakers=9`) pyannote's own count already matched, the centroids were
identical (cosine 1.000) and nothing changed. With an over-count (D, 12 people, `num_speakers=14`) the 12 real speakers' centroids were unchanged (cosine 0.998 to 1.000 against the unforced run), and pyannote added two spurious labels from fragments with 5 s and 3 s of clean speech (one of them only 42% one person). Both are under the 20 s floor, so the service returns no signature for them; the 3 s fragment scored 0.595 against the real person elsewhere, which is exactly the case the floor exists for. The extra labels still appear in `segments` as extra "Speaker N" with a few words each.

## Recommendation

- Match threshold for `pipeline-centroid-v1`: **0.60**. It sits in the middle of the measured
  gap (worst different people 0.447, worst same person 0.795), leaving about 0.15
  of margin on each side for recordings less clean than these.
- Treat 0.45 to 0.60 as "possible, ask the owner", below 0.45 as no match.
- Minimum clean speech: the service returns no signature under **20 s**
  (`VOICE_MIN_SECONDS`). Below that the evidence is thin (two speakers here)
  and method B's curve shows short samples going wrong.
- Re-measure on real meeting recordings once Coppermind has a few confirmed
  people: these are professional courtroom microphones, one room, mostly the
  same voices. Laptop microphones, Zoom codecs and crosstalk will widen both
  distributions, most likely pulling same-person scores down.

## Limits of this evaluation

- One acoustic environment (the Court), 23 distinct people, 4 recording days.
- Similar-sounding voices were present (Kagan and Sotomayor are the closest pair),
  but no family members or deliberately similar voices.
- Ground truth is Oyez's timing; its offsets are a second or so, which only
  affects the purity numbers, not the vectors.
