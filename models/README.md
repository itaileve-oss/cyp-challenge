# Model artifacts

**There are no saved model weights.** The pipeline refits from code on each run,
so nothing here is a checkpoint. What is tracked instead are the *prediction
tables* — which is what the ensemble actually needs, since every blend weight,
the TabPFN confirmation and the placement calibration are all fitted on
out-of-fold predictions rather than on the models themselves.

With these frozen, the submission can be rebuilt in seconds with no GPU and no
training.

## What is here

| pattern | contents |
|---|---|
| `cv_*_oof_predictions.csv` | out-of-fold predictions per member and feature set, one row per training compound |
| `final_ensemble_oof_predictions.csv` | the base ensemble's OOF predictions — the comparator every later step must beat |
| `final_ensemble_oof_predictions_placement_calibrated.csv` | the same after the affine placement transform |
| `final_ensemble_oof_predictions_pre_tabpfn_2026-09-24.csv` | snapshot before the TabPFN blends were added |
| `test_predictions_*.csv` | blind-set predictions per member |
| `tabpfn_confirm_results.csv` | the three-repeat, nested-weight confirmation of the TabPFN blends |
| `gd2heme_*.csv` | GalaxyDock2-HEME docking scores and reranking tables per isoform |
| `*_docking_features.csv` | docking-derived features joined to compounds |

Columns follow the challenge convention: `Molecule_Name` plus
`{CYP1A2,CYP2C9,CYP2D6,CYP3A4}_pIC50_direct_inhibition`. Blanks are genuine —
the training matrix is sparse, and 3,596 of 4,905 compounds carry only one
isoform.

## What is deliberately excluded

- **Feature and conformer caches** (~82 MB) — regenerable from
  `src/features_*.py`, and pickles are fragile across RDKit/sklearn versions
- **Docked-pose pickles** (~26 MB) — regenerable from the docking scripts

Both are listed in `.gitignore`. Regenerate with the commands in
[docs/methods/docking-features.md](../docs/methods/docking-features.md).

## A caution on reuse

The OOF predictions are tied to a specific cluster-split fold assignment. Fitting
new blend weights against them while evaluating on a *different* split will leak,
because the predictions already saw the other folds' training data. Keep the fold
definition fixed, or regenerate the OOF predictions under your own split.
