"""Tests the CYP3A4 logD feature (dimorphite-dl real pKa-based pH-7.4
correction + lipophilic-efficiency composite) on CYP1A2/CYP2C9/CYP2D6 --
unlike the docking/pharmacophore features, logD isn't isoform-specific
chemistry, it's a generic descriptor, and the dimorphite cache already
covers the full TRAIN_inhibition.csv (all isoforms), so this was never
actually untested, just never gotten to. Uses BEST hyperparams (CYP3A4's),
same repeated cluster-fold harness as everything else.
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
from features_logd import logd_feature_matrix  # noqa: E402

N_FOLDS = 5
N_REPEATS = 8
BEST = dict(num_leaves=7, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
            colsample_bytree=0.5, learning_rate=0.03, n_estimators=800,
            subsample=0.9, subsample_freq=1)
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]
ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6"]


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


def run_isoform(isoform: str, tr_all: pd.DataFrame) -> None:
    endpoint = f"{isoform}_pIC50_direct_inhibition"
    tr = tr_all[tr_all[endpoint].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[endpoint].to_numpy(dtype=float)
    print(f"\n=== {isoform}: {len(tr)} compounds ===", flush=True)

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])
    logd = logd_feature_matrix(smiles)
    X_logd = np.hstack([X_base, logd])
    n_ionizable = int(logd[:, 2].sum())
    print(f"  {n_ionizable}/{len(tr)} ionizable ({100*n_ionizable/len(tr):.1f}%)", flush=True)

    labels = ["baseline", "+logD(3)"]
    feature_sets = [X_base, X_logd]

    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    all_results = {lab: {m: [] for m in METRIC_NAMES} for lab in labels}
    st_rae_results = []

    rng = np.random.default_rng(123)
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]
        row_strae = []
        for lab, X in zip(labels, feature_sets):
            m = all_metrics(run_fold_split(X, y, relabeled, BEST), tr, endpoint)
            for k, v in m.items():
                all_results[lab][k].append(v)
            row_strae.append(m["ST-RAE"])
        print(f"  repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row_strae)), flush=True)
        st_rae_results.append(row_strae)

    st_rae_results = np.array(st_rae_results)
    print(f"\n  --- {isoform} full metric table, mean +/- std over {N_REPEATS} repeats ---")
    header = f"  {'model':12s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"  {lab:12s}"
        for m in METRIC_NAMES:
            vals = np.array(all_results[lab][m])
            row += f"{vals.mean():>8.4f}±{vals.std():.3f}".rjust(14)
        print(row)

    d = st_rae_results[:, 1] - st_rae_results[:, 0]
    t, p = ttest_rel(st_rae_results[:, 1], st_rae_results[:, 0])
    print(f"  {isoform} +logD vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  "
          f"t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


def main() -> None:
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    for iso in ISOFORMS:
        run_isoform(iso, tr_all)


if __name__ == "__main__":
    main()
