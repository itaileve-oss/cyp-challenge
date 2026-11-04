# Crystal contact-motif scoring

A knowledge-based pose score built from deposited CYP3A4 complexes. Included
with a warning: it is the metric that drove our first submission, and that
submission underperformed a trivial baseline.

## Building the reference

From 100 known CYP3A4–ligand complexes, 96 of them active-site (4 are peripheral
— progesterone at 17 Å in 1W0F/5A1P/5A1R, and 8DYC at 12 Å — and must be
excluded):

| measurement | value |
|---|---|
| Fe–ligand distance, all active-site | median 2.44 Å (IQR 2.29–2.60) |
| Fe–ligand distance, true N-coordinating only | **median 2.35 Å (IQR 2.22–2.47)** |
| closest atom | N in 84/96, C in 11, O in 1 |
| contacted residues | median 16 |
| tilt from heme normal | median 8.1° (coordinating) |

**Use the coordinating-only figure.** Our first analysis quoted 2.44 Å, which is
inflated by non-coordinators sitting 3–6 Å out, and that led to an overstated
claim that co-folding was 0.4 Å too short. The real gap is ~0.2 Å.

The consensus motif is the 14 residues contacted by ≥50% of active-site poses:

```
Ser119 98%  Thr309 96%  Arg212 96%  Ile369 96%  Ala305 95%
Ile301 89%  Phe213 84%  Arg105 82%  Phe108 81%  Ala370 80%
Phe215 78%  Phe241 77%  Ile120 74%  Phe304 66%
```

## Scoring

```bash
python structure/src/motif_score.py POSE_CSV OUT_CSV --valid-only --workers 90
```

`motif_weighted` is the frequency-weighted fraction of those 14 residues a pose
contacts within 4.5 Å, so missing Ser119 (98%) costs more than missing Phe304
(66%). Also reported: raw recall, contact count, and whether the Fe distance
falls in the crystal IQR.

## Known weaknesses — read before using it

**It is recall-only.** Contacting residues outside the motif is never penalised,
so the score rises with burial. A bigger or more deeply buried pose scores higher
almost by construction.

**It is blind to orientation.** A ligand flipped end-for-end contacts the same
residues and scores identically.

**It was never validated against held-out truth.** We used it to pick 16 of 20
poses on the strength of its being "experimentally grounded", which conflates
*derived from experiment* with *predictive of accuracy*. The resulting entry
scored LDDT-PLI 0.371 where the top-confidence baseline scores ~0.414.

## Where it is genuinely useful

As a diagnostic rather than a selector. It showed that our poses contact a median
of 12 residues against the crystal median of 16 — a systematic under-burial
traceable to the rigid predicted receptor, which no amount of pose selection can
fix. That finding came from this score and is worth having.
