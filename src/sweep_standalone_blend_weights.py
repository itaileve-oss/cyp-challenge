"""Sweep the blend weight for adding a standalone single-task OOF component
(CYP3A4 docking+pharma+logD, CYP2D6 pharmacophore) into the existing base
ensemble (models/final_ensemble_oof_predictions.csv), per-isoform only --
same pattern as the original (now-removed) CYP3A4 pharmacophore blend sweep
(0.1-0.5, best=0.35). Picks the weight minimizing that isoform's own ST-RAE
on the local OOF set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)
from cv_multitask import CYP_ISOFORMS  # noqa: E402

UNIMOL_WEIGHT = 0.1

COMPONENTS = {
    "CYP3A4": "cv_cyp3a4_docking_pharma_logd_singletask_oof_predictions.csv",
    "CYP2D6": "cv_cyp2d6_final_singletask_oof_predictions.csv",
    "CYP1A2": "cv_cyp1a2_final_singletask_oof_predictions.csv",
}
WEIGHTS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def st_rae_for_isoform(oof: np.ndarray, tr: pd.DataFrame, isoform: str) -> float:
    endpoint = f"{isoform}_pIC50_direct_inhibition"
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], endpoint: oof})
    ground_truth = tr[["Molecule_Name", endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [endpoint])
    track = add_macro_endpoint(boot, [endpoint], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return float(summary.loc[endpoint, "ST-RAE_mean"])


def main() -> None:
    models_dir = PROJECT_ROOT / "models"
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    mol_order = tr["Molecule_Name"]

    # pure, unblended base ensemble (GNN+LGBM+UniMol only) -- NOT
    # final_ensemble_oof_predictions.csv, which already has standalone blends
    # baked in from a previous sweep and would double-count them here
    gnn = pd.read_csv(models_dir / "cv_gnn_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order]
    lgbm = pd.read_csv(models_dir / "cv_multitask_tuned_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order]
    unimol = pd.read_csv(models_dir / "cv_unimol_only_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order]
    cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    base_ens = (0.5 * gnn[cols] + 0.5 * lgbm[cols]).reset_index(drop=True)
    base_ens = (1 - UNIMOL_WEIGHT) * base_ens + UNIMOL_WEIGHT * unimol[cols].reset_index(drop=True)
    base_ens.index = mol_order.values
    base = base_ens

    best_weights = {}
    for isoform, fname in COMPONENTS.items():
        endpoint = f"{isoform}_pIC50_direct_inhibition"
        tr_iso = tr[tr[endpoint].notna()].reset_index(drop=True)
        standalone = pd.read_csv(models_dir / fname).set_index("Molecule_Name")

        base_pred = base.loc[tr_iso["Molecule_Name"], endpoint].to_numpy(dtype=float)
        standalone_pred = standalone.loc[tr_iso["Molecule_Name"], endpoint].to_numpy(dtype=float)

        print(f"\n=== {isoform}: sweeping blend weight ===")
        results = []
        for w in WEIGHTS:
            blended = (1 - w) * base_pred + w * standalone_pred
            score = st_rae_for_isoform(blended, tr_iso, isoform)
            results.append((w, score))
            print(f"  w={w:.2f}  ST-RAE={score:.4f}")

        best_w, best_score = min(results, key=lambda r: r[1])
        base_score = dict(results)[0.0]
        print(f"  -> best w={best_w:.2f} (ST-RAE={best_score:.4f}, vs w=0 base={base_score:.4f}, "
              f"delta={best_score-base_score:+.4f})")
        best_weights[isoform] = best_w

    print(f"\n=== Summary ===")
    for isoform, w in best_weights.items():
        print(f"{isoform}: best_weight={w}")


if __name__ == "__main__":
    main()
