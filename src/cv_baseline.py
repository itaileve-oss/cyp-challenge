"""Scaffold-split local CV baseline for the Direct Inhibition regression track.

Trains one single-task LightGBM model per isoform, out-of-fold, on Bemis-Murcko
scaffold groups (so near-identical scaffolds never split across train/test within a
fold, matching the analog-expansion structure of the real blind test set). Scores the
resulting out-of-fold predictions with the challenge's own ported evaluation code
(evaluation/evaluate_predictions.py) so the local number tracks the real leaderboard
metric (MA-ST-RAE), not a proxy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))

from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

from features import morgan_fp_matrix  # noqa: E402

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
N_FOLDS = 5
SEED = 42


def run_isoform_oof(df: pd.DataFrame, isoform: str, fp: np.ndarray) -> np.ndarray:
    """Out-of-fold LGBM predictions for one isoform, NaN where the label is missing."""
    label_col = f"{isoform}_pIC50_direct_inhibition"
    has_label = df[label_col].notna().to_numpy()
    oof = np.full(len(df), np.nan)

    idx = np.flatnonzero(has_label)
    y = df.loc[has_label, label_col].to_numpy()
    X = fp[has_label]
    groups = uru.get_bemis_murcko_clusters(df.loc[has_label, "SMILES"].tolist())

    n_groups = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = LGBMRegressor(random_state=SEED, verbosity=-1)
        model.fit(X[train_idx], y[train_idx])
        oof[idx[test_idx]] = model.predict(X[test_idx])

    return oof


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    fp = morgan_fp_matrix(tr["SMILES"].tolist())

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for isoform in CYP_ISOFORMS:
        print(f"Running OOF CV for {isoform}...", flush=True)
        oof = run_isoform_oof(tr, isoform, fp)
        predictions[f"{isoform}_pIC50_direct_inhibition"] = oof
        n_scored = np.isfinite(oof).sum()
        print(f"  {isoform}: {n_scored} out-of-fold predictions")

    ground_truth_cols = ["Molecule_Name"] + [
        c
        for c in tr.columns
        if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    ground_truth = tr[ground_truth_cols]

    bootstrap_results = score_activity_predictions(predictions, ground_truth, REGRESSION_ENDPOINTS)
    track_results = add_macro_endpoint(bootstrap_results, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track_results)

    metric_cols = [f"{name}_mean" for name, _ in ACTIVITY_METRICS] + [
        f"{name}_std" for name, _ in ACTIVITY_METRICS
    ]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Local scaffold-split CV, scored with real evaluation code ===")
    print(summary[metric_cols].round(4))

    out_path = PROJECT_ROOT / "models" / "cv_baseline_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\nSaved OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
