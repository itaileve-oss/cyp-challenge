"""Manual random hyperparameter search for LightGBM on the current best CYP3A4
feature set (fp+desc + docking(3) + pharma(5) + logD(3) = 11 extra columns),
searching around the hand-picked TUNED config used everywhere else this
session (num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
colsample_bytree=0.7, n_estimators=300, learning_rate=0.03).

Selection objective: mean ST-RAE over N_SEARCH_REPEATS randomized cluster-fold
splits (not a single split -- this project's established rigor standard says
a single split isn't trustworthy for picking a "winner", so the search itself
uses a small repeated-CV average, then the winning config gets the full
N_FINAL_REPEATS validation against the existing TUNED baseline before being
trusted).
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
from cv_repeated_combined_docking_pharma import build_docking, build_pharma_nan  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
N_FOLDS = 5
N_SEARCH_REPEATS = 3
N_FINAL_REPEATS = 8
N_TRIALS = 40

TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)

METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]


def all_metrics(oof: np.ndarray, tr: pd.DataFrame) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [ENDPOINT])
    track = add_macro_endpoint(boot, [ENDPOINT], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[ENDPOINT, f"{m}_mean"]) for m in METRIC_NAMES}


def st_rae(oof: np.ndarray, tr: pd.DataFrame) -> float:
    return all_metrics(oof, tr)["ST-RAE"]


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params: dict) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1, **params)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def sample_params(rng: np.random.Generator) -> dict:
    return dict(
        num_leaves=int(rng.choice([7, 15, 31, 63])),
        min_child_samples=int(rng.choice([20, 50, 100])),
        reg_alpha=float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0])),
        reg_lambda=float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0])),
        colsample_bytree=float(rng.choice([0.5, 0.7, 0.9, 1.0])),
        learning_rate=float(rng.choice([0.01, 0.03, 0.05, 0.1])),
        n_estimators=int(rng.choice([200, 300, 500, 800])),
        subsample=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        subsample_freq=1,
    )


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
    X = np.hstack([fp, desc, dock, pharma, logd])
    print(f"X shape: {X.shape}", flush=True)

    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    # fixed set of search-phase cluster relabelings, shared across all trials
    search_rng = np.random.default_rng(7)
    search_relabelings = []
    for _ in range(N_SEARCH_REPEATS):
        perm = search_rng.permutation(len(unique_clusters))
        search_relabelings.append(perm[cluster_int])

    trial_rng = np.random.default_rng(999)
    results = []

    # trial 0: the existing TUNED config, as a reference point in the same search loop
    configs = [("TUNED (existing)", TUNED)]
    for t in range(N_TRIALS):
        configs.append((f"trial{t}", sample_params(trial_rng)))

    for name, params in configs:
        scores = [st_rae(run_fold_split(X, y, cl, params), tr) for cl in search_relabelings]
        mean_score = float(np.mean(scores))
        results.append((name, mean_score, params))
        print(f"{name}: mean ST-RAE (n={N_SEARCH_REPEATS})={mean_score:.4f}  params={params}", flush=True)

    results.sort(key=lambda r: r[1])
    print("\n=== Top 5 configs by search-phase mean ST-RAE ===")
    for name, score, params in results[:5]:
        print(f"{name}: {score:.4f}  {params}")

    best_name, best_score, best_params = results[0]
    print(f"\nBest found: {best_name} ({best_score:.4f}). Running full {N_FINAL_REPEATS}-repeat "
          f"validation against TUNED (existing)...", flush=True)

    final_rng = np.random.default_rng(123)  # same seed used everywhere else this session
    st_rae_pairs = []
    for rep in range(N_FINAL_REPEATS):
        perm = final_rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]
        s_tuned = st_rae(run_fold_split(X, y, relabeled, TUNED), tr)
        s_best = st_rae(run_fold_split(X, y, relabeled, best_params), tr)
        st_rae_pairs.append((s_tuned, s_best))
        print(f"repeat {rep}: TUNED(existing)={s_tuned:.4f}  best_found={s_best:.4f}", flush=True)

    arr = np.array(st_rae_pairs)
    d = arr[:, 1] - arr[:, 0]
    t, p = ttest_rel(arr[:, 1], arr[:, 0])
    print(f"\nbest_found vs TUNED(existing): delta={d.mean():+.4f} std={d.std():.4f}  "
          f"t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_FINAL_REPEATS}")
    print(f"best_params = {best_params}")


if __name__ == "__main__":
    main()
