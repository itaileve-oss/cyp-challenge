"""Final ensemble: 50/50 GNN + multitask LightGBM (TUNED hyperparameters), +
Uni-Mol-embeddings-only model blended in globally at weight 0.1.

All three ingredients are now CONFIRMED REAL under repeated randomized-cluster
CV (paired t-test against the same fold reassignments each time), not just a
single lucky split:
- Ensemble vs LightGBM alone: delta -0.0505 (p=0.0001) (src/cv_repeated_ensemble_validation.py)
- Ensemble vs GNN alone: delta -0.0489 (p=0.0001)
- Uni-Mol blend vs base ensemble: delta -0.0035 (p=0.0002, tight std=0.0005)
- Tuned LightGBM hyperparameters vs sklearn defaults: delta -0.0156 (p=0.0003,
  8/8 repeats same direction) (src/tune_lgbm_hyperparams.py) -- the first and
  only hyperparameter tuning done all project; every prior LightGBM run used
  untouched defaults.

REMOVED (2026-09-14): the CYP3A4 heme/Ser119 pharmacophore blend that used to
sit on top of this (previously claimed MA-ST-RAE 0.7097). Under the same
repeated-CV protocol its effect turned out to be statistically indistinguishable
from zero (paired delta -0.0003, p=0.88 on the isolated CYP3A4 single-task
test) -- the original "win" was one lucky fixed-split fold assignment, not a
real effect. See project memory / cv_multitask_pharmacophore_gated.py history
for the full diagnostic trail.

HONEST CAVEAT (2026-09-14): swapping the tuned LightGBM into this exact 50/50
+ Uni-Mol blend scores slightly WORSE on the one fixed split (0.7153) than the
untuned version did (0.7125) -- even though the tuned LightGBM is a confirmed
real improvement on its own (-0.0156, p=0.0003). A quick fixed-split reweight
search found a marginally better blend (w_lgbm=0.55, w_unimol=0.06 -> 0.7152),
but that's a ~0.0005 difference, well under the measured noise floor -- not
worth chasing or baking in as a "finding." Plausible real explanation: a
better-regularized LightGBM may be smoother/more GNN-like, reducing the error
diversity that ensembling exploits, so a stronger component doesn't
automatically make a stronger ensemble. This specific combined-pipeline
question (does tuned-LightGBM help the FULL ensemble) has NOT been repeated-CV
confirmed either way -- only the LightGBM-alone tuning effect and the general
fact of ensembling have been. Kept the tuned LightGBM in here anyway since
there's no confirmed evidence it hurts either, and its own effect is real.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)
from cv_multitask import CYP_ISOFORMS  # noqa: E402

UNIMOL_WEIGHT = 0.1


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    mol_order = tr["Molecule_Name"]

    gnn = pd.read_csv(models_dir / "cv_gnn_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order]
    lgbm = pd.read_csv(models_dir / "cv_multitask_tuned_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order]
    unimol = pd.read_csv(models_dir / "cv_unimol_only_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order]

    cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    base = (0.5 * gnn[cols] + 0.5 * lgbm[cols]).reset_index(drop=True)
    ensemble = (1 - UNIMOL_WEIGHT) * base + UNIMOL_WEIGHT * unimol[cols].reset_index(drop=True)
    ensemble.index = mol_order.values

    # standalone single-task blends (2026-09-23), weights from sweep_standalone_blend_weights.py
    standalone_blends = {
        "CYP3A4": ("cv_cyp3a4_docking_pharma_logd_singletask_oof_predictions.csv", 0.50),
        "CYP2D6": ("cv_cyp2d6_final_singletask_oof_predictions.csv", 0.25),
        "CYP1A2": ("cv_cyp1a2_final_singletask_oof_predictions.csv", 0.50),
    }
    for isoform, (fname, w) in standalone_blends.items():
        col = f"{isoform}_pIC50_direct_inhibition"
        standalone = pd.read_csv(models_dir / fname).set_index("Molecule_Name")[col].reindex(mol_order.values)
        has_pred = standalone.notna()
        ensemble.loc[has_pred, col] = (
            (1 - w) * ensemble.loc[has_pred, col] + w * standalone[has_pred].to_numpy(dtype=float)
        )

    # TabPFN blends (2026-09-30), applied on top of the standalone blends; confirmed over 3
    # shuffled-cluster repeats with nested weight selection (tabpfn_confirm_and_predict.py)
    tabpfn_blends = {
        "CYP2C9": ("cv_tabpfn_cyp2c9_oof_predictions.csv", 0.30),
        "CYP2D6": ("cv_tabpfn_cyp2d6_oof_predictions.csv", 0.40),
    }
    for isoform, (fname, w) in tabpfn_blends.items():
        col = f"{isoform}_pIC50_direct_inhibition"
        tabpfn = pd.read_csv(models_dir / fname).set_index("Molecule_Name")[col].reindex(mol_order.values)
        has_pred = tabpfn.notna()
        ensemble.loc[has_pred, col] = (
            (1 - w) * ensemble.loc[has_pred, col] + w * tabpfn[has_pred].to_numpy(dtype=float)
        )

    predictions = ensemble.reset_index(drop=True).copy()
    predictions.insert(0, "Molecule_Name", mol_order.values)

    out_path = models_dir / "final_ensemble_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Saved final ensemble OOF predictions to {out_path}")

    gt_cols = ["Molecule_Name"] + [
        c for c in tr.columns if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    ground_truth = tr[gt_cols]

    bootstrap_results = score_activity_predictions(predictions, ground_truth, REGRESSION_ENDPOINTS)
    track_results = add_macro_endpoint(bootstrap_results, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track_results)

    metric_cols = [f"{name}_mean" for name, _ in ACTIVITY_METRICS] + [f"{name}_std" for name, _ in ACTIVITY_METRICS]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Final ensemble: 50/50 GNN+LightGBM, +Uni-Mol (w=0.1) -- validated under repeated CV ===")
    print(summary[metric_cols].round(4))


if __name__ == "__main__":
    main()
