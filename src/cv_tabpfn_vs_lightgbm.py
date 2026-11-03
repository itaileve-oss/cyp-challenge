"""TabPFN (compact feature set: RDKit descriptors + docking(3) + pharma(5) +
logD(3), no Morgan fingerprint -- fingerprints are ~4x slower for TabPFN and
likely a poor fit for its pretrained continuous-tabular prior) vs. today's
two LightGBM benchmarks, same repeated cluster-fold harness and full metric
set as every other comparison this session.

TabPFN is dramatically slower per fold than LightGBM (~7min vs ~seconds), so
this uses N_REPEATS=3 instead of the usual 8 -- a first-pass read, not the
full rigor standard. Run in background; expect ~2 hours.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import RDLogger
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor

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
from cv_repeated_combined_docking_pharma import build_docking, build_pharma_nan  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
N_FOLDS = 5
N_REPEATS = 3

TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)
BEST = dict(num_leaves=7, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
            colsample_bytree=0.5, learning_rate=0.03, n_estimators=800,
            subsample=0.9, subsample_freq=1)

METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]


def all_metrics(oof: np.ndarray, tr: pd.DataFrame) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [ENDPOINT])
    track = add_macro_endpoint(boot, [ENDPOINT], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[ENDPOINT, f"{m}_mean"]) for m in METRIC_NAMES}


def run_lgbm_fold(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params: dict) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1, **params)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def run_tabpfn_fold(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=cluster_labels)):
        t0 = time.time()
        model = TabPFNRegressor(device="cuda", random_state=42)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
        print(f"    tabpfn fold {i}: {time.time()-t0:.1f}s", flush=True)
    return oof


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    dock = build_docking(tr)
    pharma = build_pharma_nan(tr)
    logd = logd_feature_matrix(smiles)
    X_base = np.hstack([fp, desc])
    X_all11 = np.hstack([fp, desc, dock, pharma, logd])
    X_compact = np.hstack([desc, dock, pharma, logd])
    print(f"X_base={X_base.shape} X_all11={X_all11.shape} X_compact={X_compact.shape}", flush=True)

    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    labels = ["baseline (LGBM/TUNED)", "all11 (LGBM/BEST)", "compact (TabPFN)"]
    all_results = {lab: {m: [] for m in METRIC_NAMES} for lab in labels}

    rng = np.random.default_rng(123)
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]
        print(f"\n=== repeat {rep} ===", flush=True)

        oof_baseline = run_lgbm_fold(X_base, y, relabeled, TUNED)
        m_baseline = all_metrics(oof_baseline, tr)

        oof_best = run_lgbm_fold(X_all11, y, relabeled, BEST)
        m_best = all_metrics(oof_best, tr)

        print("  running TabPFN (compact)...", flush=True)
        oof_tabpfn = run_tabpfn_fold(X_compact, y, relabeled)
        m_tabpfn = all_metrics(oof_tabpfn, tr)

        for lab, m in zip(labels, [m_baseline, m_best, m_tabpfn]):
            for k, v in m.items():
                all_results[lab][k].append(v)
        print(f"repeat {rep}: " + "  ".join(
            f"{lab}(ST-RAE)={m['ST-RAE']:.4f}" for lab, m in zip(labels, [m_baseline, m_best, m_tabpfn])
        ), flush=True)

    print(f"\n=== Full metric table, mean +/- std over {N_REPEATS} repeats ===")
    header = f"{'model':24s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"{lab:24s}"
        for m in METRIC_NAMES:
            vals = np.array(all_results[lab][m])
            row += f"{vals.mean():>8.4f}±{vals.std():.3f}".rjust(14)
        print(row)


if __name__ == "__main__":
    main()
