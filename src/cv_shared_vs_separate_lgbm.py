"""Tests the user's hypothesis directly: does a SEPARATE single-task LightGBM
per isoform (own hyperparameters, own model) beat the current shared
multitask LightGBM (one model, one hyperparameter config, trained on all 4
isoforms stacked together) -- for CYP1A2 and CYP2C9, the two isoforms with no
standalone component yet. Isolates the "shared vs separate architecture"
question from any feature-engineering question (same fp+desc features both
sides) and from the BEST-vs-TUNED question (uses TUNED, since we just
confirmed BEST doesn't transfer off CYP3A4).

Compares against the ACTUAL shared multitask model's own OOF predictions
(cv_multitask_tuned_oof_predictions.csv), not a re-derived baseline, so this
is a true apples-to-apples "current architecture vs proposed architecture"
comparison, not two different re-implementations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import RDLogger
from scipy.stats import ttest_rel
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import useful_rdkit_utils as uru  # noqa: E402
from evaluation.config import ACTIVITY_METRICS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402

N_FOLDS = 5
N_REPEATS = 8
TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]
ISOFORMS = ["CYP1A2", "CYP2C9"]


def all_metrics(oof: np.ndarray, tr: pd.DataFrame, endpoint: str) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], endpoint: oof})
    ground_truth = tr[["Molecule_Name", endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [endpoint])
    track = add_macro_endpoint(boot, [endpoint], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[endpoint, f"{m}_mean"]) for m in METRIC_NAMES}


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params: dict) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1, **params)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def main() -> None:
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    shared_oof = pd.read_csv(PROJECT_ROOT / "models" / "cv_multitask_tuned_oof_predictions.csv").set_index("Molecule_Name")

    for isoform in ISOFORMS:
        endpoint = f"{isoform}_pIC50_direct_inhibition"
        tr = tr_all[tr_all[endpoint].notna()].reset_index(drop=True)
        smiles = tr["SMILES"].tolist()
        y = tr[endpoint].to_numpy(dtype=float)
        print(f"\n=== {isoform}: {len(tr)} compounds ===", flush=True)

        # the ACTUAL shared multitask model's own OOF, scored on this isoform's rows
        shared_pred = shared_oof.loc[tr["Molecule_Name"], endpoint].to_numpy(dtype=float)
        m_shared = all_metrics(shared_pred, tr, endpoint)
        print(f"  shared multitask (current): ST-RAE={m_shared['ST-RAE']:.4f}", flush=True)

        fp = morgan_fp_matrix(smiles)
        desc = rdkit_descriptor_matrix(smiles)
        X = np.hstack([fp, desc])

        base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
        unique_clusters = np.unique(base_clusters)
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

        separate_scores = []
        rng = np.random.default_rng(123)
        for rep in range(N_REPEATS):
            perm = rng.permutation(len(unique_clusters))
            relabeled = perm[cluster_int]
            oof = run_fold_split(X, y, relabeled, TUNED)
            m = all_metrics(oof, tr, endpoint)
            separate_scores.append(m["ST-RAE"])
            print(f"  repeat {rep}: separate single-task ST-RAE={m['ST-RAE']:.4f}", flush=True)

        separate_scores = np.array(separate_scores)
        shared_scores = np.full(N_REPEATS, m_shared["ST-RAE"])  # shared model has no repeat variance (fixed OOF)
        d = separate_scores - shared_scores
        t, p = ttest_rel(separate_scores, shared_scores)
        print(f"\n  {isoform}: separate mean={separate_scores.mean():.4f}±{separate_scores.std():.3f}  "
              f"shared={m_shared['ST-RAE']:.4f}  delta={d.mean():+.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


if __name__ == "__main__":
    main()
