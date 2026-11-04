# Regression ensemble

Predicts pIC50 for CYP1A2/2C9/2D6/3A4. Scored by MA-ST-RAE: the macro-average of
a soft-threshold relative absolute error that charges zero inside each label's
credible interval and the distance to the nearer bound outside it.

Local MA-ST-RAE **0.6921**; live macro ST-RAE **0.6622** on the half-test
leaderboard. Local is pessimistic — see
[placement-calibration.md](placement-calibration.md).

## Base ensemble

Only 6,525 dose-response labels exist across 4,905 compounds, and 3,596 of those
compounds carry just one isoform. Sharing a representation across isoforms
therefore matters more than any single architecture, so the base is a multitask
blend:

| weight | member |
|---|---|
| 0.45 | message-passing GNN on molecular topology |
| 0.45 | LightGBM on descriptors + fingerprints, multitask across isoforms |
| 0.10 | model on Uni-Mol 3D embeddings |

The Uni-Mol component is small but earned its place on a consistent gain
(ΔST-RAE −0.0035, p = 0.0002) rather than on standalone accuracy.

```bash
python src/build_final_ensemble.py        # fit, write OOF predictions
python src/build_test_submission.py       # apply to the blind set, calibrate
```

## Per-isoform additions

Each isoform gets a single-task model blended on top at a swept weight, carrying
features specific to that enzyme's binding chemistry:

| isoform | added signal | weight |
|---|---|---|
| CYP3A4 | docking coordination features; Kaur-style pharmacophore measured from the heme-coordinating atom; logD with pKa-based ionisation | 0.50 |
| CYP1A2 | docking; four-point pharmacophore — flat aromatic ring, acceptor, hydrophobe, **and a donor** beyond the literature's three-point description | 0.50 |
| CYP2D6 | docking aggregated best-across-receptors; basic-amine to aromatic-ring distance | 0.25 |
| CYP2C9 | none adopted — see below | — |

The CYP1A2 four-point pharmacophore was the largest single-isoform gain in the
project, and the hydrogen-bond donor was the part that mattered: the three-point
version alone did nothing.

## CYP2C9 resisted everything

Sixteen distinct approaches were tried and confirmed null before TabPFN: docking
under two aggregation strategies, acidic-group presence and distance, a
generalised azole/pyridine/sulfonamide coordinating-group definition, an
Arg108-anchored distance, the CYP1A2 pharmacophore template, logD, strain
energy, 3D shape descriptors, L1 loss, and confidence-interval-weighted
training.

The likely explanation is in the labels: **CYP2C9 has the widest credible
intervals relative to its own label spread of all four isoforms** — roughly a
90% CI-width to standard-deviation ratio against 54–80% elsewhere. That looks
like a data-noise ceiling rather than a missing descriptor.

## Validation

Repeated cluster-split cross-validation throughout. Blend weights are chosen by
sweep, and confirmed by **nested** selection wherever a gain is small enough for
selection optimism to matter. All six reported metrics (ST-RAE, MAE, R², Pearson,
Spearman, Kendall) are checked together, not ST-RAE alone — a change that
improves the scored metric while degrading correlation is not adopted.

See [tabpfn-blend.md](tabpfn-blend.md) and
[docking-features.md](docking-features.md).
