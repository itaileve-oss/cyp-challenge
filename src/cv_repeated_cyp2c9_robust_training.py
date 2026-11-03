"""Two training-side (not feature-side) approaches for CYP2C9, targeting the
noise-ceiling finding directly instead of searching for more structural
signal: (1) L1/MAE loss instead of L2 -- less dominated by noisy-label
outliers; (2) sample weighting by inverse confidence-interval width -- lets
the model trust well-measured compounds more than noisy ones. Baseline
(fp+desc, L2) vs each, TUNED hyperparams otherwise.
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

ENDPOINT = "CYP2C9_pIC50_direct_inhibition"
N_FOLDS = 5
N_REPEATS = 8
TUNED_L2 = dict(objective="regression", num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
                colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)
TUNED_L1 = dict(TUNED_L2, objective="regression_l1")
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]


def all_metrics(oof: np.ndarray, tr: pd.DataFrame) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [ENDPOINT])
    track = add_macro_endpoint(boot, [ENDPOINT], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[ENDPOINT, f"{m}_mean"]) for m in METRIC_NAMES}


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params: dict,
                    sample_weight: np.ndarray | None = None) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1, **params)
        sw = sample_weight[tr_idx] if sample_weight is not None else None
        model.fit(X[tr_idx], y[tr_idx], sample_weight=sw)
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def main() -> None:
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr_all[tr_all[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)
    print(f"CYP2C9: {len(tr)} compounds", flush=True)

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X = np.hstack([fp, desc])

    ci_width = (tr[f"{ENDPOINT}_conf_high"] - tr[f"{ENDPOINT}_conf_low"]).to_numpy(dtype=float)
    ci_width = np.clip(ci_width, 0.05, None)  # avoid div-by-zero / extreme weights
    inv_ci_weight = 1.0 / ci_width
    inv_ci_weight = inv_ci_weight / inv_ci_weight.mean()  # normalize to mean 1
    print(f"  CI-weight range: {inv_ci_weight.min():.2f}-{inv_ci_weight.max():.2f}, "
          f"median={np.median(inv_ci_weight):.2f}", flush=True)

    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    labels = ["baseline (L2)", "L1/MAE loss", "L2 + CI-weighted"]
    all_results = {lab: {m: [] for m in METRIC_NAMES} for lab in labels}
    st_rae_results = []

    rng = np.random.default_rng(123)
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]
        row = []

        oof_l2 = run_fold_split(X, y, relabeled, TUNED_L2)
        oof_l1 = run_fold_split(X, y, relabeled, TUNED_L1)
        oof_ciw = run_fold_split(X, y, relabeled, TUNED_L2, sample_weight=inv_ci_weight)

        for lab, oof in zip(labels, [oof_l2, oof_l1, oof_ciw]):
            m = all_metrics(oof, tr)
            for k, v in m.items():
                all_results[lab][k].append(v)
            row.append(m["ST-RAE"])
        print(f"repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row)), flush=True)
        st_rae_results.append(row)

    st_rae_results = np.array(st_rae_results)
    print(f"\n=== Full metric table, mean +/- std over {N_REPEATS} repeats ===")
    header = f"{'model':20s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"{lab:20s}"
        for m in METRIC_NAMES:
            vals = np.array(all_results[lab][m])
            row += f"{vals.mean():>8.4f}±{vals.std():.3f}".rjust(14)
        print(row)

    print()
    for i, lab in enumerate(labels[1:], start=1):
        d = st_rae_results[:, i] - st_rae_results[:, 0]
        t, p = ttest_rel(st_rae_results[:, i], st_rae_results[:, 0])
        print(f"{lab} vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


if __name__ == "__main__":
    main()
