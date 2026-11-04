# Placement calibration

A per-isoform affine transform applied to the final predictions. It cannot change
any compound's rank — only where the predicted distribution sits and how wide it
is.

```bash
python src/apply_placement_calibration.py
```

## Why spread matters under this metric

ST-RAE charges zero inside the label's credible interval and the distance to the
nearer bound outside it. A well-ranked but **compressed** column therefore scores
badly: predictions collapse toward the centre and stop landing inside intervals
out in the tails, where the intervals are no wider. Blending compresses spread by
construction, since it averages toward the mean of its members.

The transform rescales each isoform's predictions about its own mean to a target
standard deviation of `rho × sd_blind`, then shifts to the blind-set mean.

## Solving for rho from leaderboard feedback

Initially `rho` was the local out-of-fold Pearson. The interim leaderboard
allowed solving for the **true blind correlation** directly from the reported R²:

```
rho_true = (R²_blind + k²) / (2k)      # k = local OOF Pearson at submission time
```

applied thereafter as a fixed per-isoform inflation factor
(`rho_true_today = INFLATION[iso] × rho_local_today`):

| isoform | rho local | rho true | inflation |
|---|---|---|---|
| CYP1A2 | 0.553 | 0.703 | 1.27× |
| CYP2C9 | 0.657 | 0.782 | 1.19× |
| CYP2D6 | 0.423 | 0.630 | 1.49× |
| CYP3A4 | 0.788 | 0.756 | 0.96× |

Three isoforms were **pessimistic** under cluster-split CV — their true blind
correlation is higher than local validation suggested, so their spread had been
under-scaled. CYP3A4 was mildly optimistic, which explains a regression on an
earlier submission where the uncorrected scale over-stretched it. With the old
calibration CYP3A4's observed blind R² of 0.574 was already within 0.001 of its
theoretical ceiling, so the scale *maths* was never the bug — the input `rho`
was.

## What this does not fix, and cannot

The transform corrects **spread, not location**. Out of fold the optimal additive
offset is zero at every endpoint *by construction*: cross-validation holds the
train-to-test gap constant, and a systematic shift between the training and blind
populations is exactly what it cannot see.

This is not a minor omission. Another entrant reported that a **uniform additive
offset per endpoint was the single largest improvement in their entire project**,
worth more than any modelling change — two constants read off a leaderboard
(+0.170 and +0.140 on two endpoints, 0.4436 → 0.4283). Finding them requires
spending a submission to measure location, which we never did.

## Related: why training labels are shifted

Dose-response pIC50 was measured only for compounds that hit in the
single-concentration screen, so the training labels are enriched for strong
inhibitors while the blind set is not — worst for CYP1A2 and CYP2D6. The
principled fix is to convert single-concentration log2 fold-change into pIC50
**intervals** (inactives from the assay floor to ~3.2–3.4; the rest binned by
log2fc, each bin taking the 2.5–97.5 percentile range of confirmed compounds),
which recovers the weak compounds and reduces how much work calibration has to
do. `single_conc_TRAIN.csv` carries the data; we used it only as a feature.
