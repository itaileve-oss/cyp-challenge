"""Closes the hyperparameter gap that hid CYP2D6's real docking signal:
CYP1A2's docking(3) was only ever tested under BEST (CYP3A4's hyperparams);
CYP2C9's acidic-group pharmacophore was only ever tested under plain sklearn
defaults. Retests both under TUNED -- the config just confirmed correct for
CYP2D6 -- before investing in any new feature engineering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, RDLogger
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
from features_pharmacophore_other_isoforms import cyp2c9_feature_matrix  # noqa: E402
from cv_repeated_docking_other_isoforms import build_docking_isoform  # noqa: E402

N_FOLDS = 5
N_REPEATS = 8
TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]


def all_metrics(oof: np.ndarray, tr: pd.DataFrame, endpoint: str) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], endpoint: oof})
    ground_truth = tr[["Molecule_Name", endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [endpoint])
    track = add_macro_endpoint(boot, [endpoint], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[endpoint, f"{m}_mean"]) for m in METRIC_NAMES}


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1, **TUNED)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def repeated_test(labels, feature_sets, y, tr, endpoint, base_clusters):
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    all_results = {lab: {m: [] for m in METRIC_NAMES} for lab in labels}
    st_rae_results = []
    rng = np.random.default_rng(123)
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]
        row = []
        for lab, X in zip(labels, feature_sets):
            m = all_metrics(run_fold_split(X, y, relabeled), tr, endpoint)
            for k, v in m.items():
                all_results[lab][k].append(v)
            row.append(m["ST-RAE"])
        print(f"  repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row)), flush=True)
        st_rae_results.append(row)
    st_rae_results = np.array(st_rae_results)

    print(f"\n  --- full metric table, TUNED hyperparams ---")
    header = f"  {'model':14s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"  {lab:14s}"
        for m in METRIC_NAMES:
            vals = np.array(all_results[lab][m])
            row += f"{vals.mean():>8.4f}±{vals.std():.3f}".rjust(14)
        print(row)
    for i, lab in enumerate(labels[1:], start=1):
        d = st_rae_results[:, i] - st_rae_results[:, 0]
        t, p = ttest_rel(st_rae_results[:, i], st_rae_results[:, 0])
        print(f"  {lab} vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


def main() -> None:
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")

    print("=== CYP1A2: docking(3) under TUNED ===", flush=True)
    endpoint = "CYP1A2_pIC50_direct_inhibition"
    tr = tr_all[tr_all[endpoint].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[endpoint].to_numpy(dtype=float)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])
    dock = build_docking_isoform("CYP1A2", tr)
    X_dock = np.hstack([X_base, dock])
    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    repeated_test(["baseline", "+docking(3)"], [X_base, X_dock], y, tr, endpoint, base_clusters)

    print("\n=== CYP2C9: acidic-group pharmacophore(2) under TUNED ===", flush=True)
    endpoint = "CYP2C9_pIC50_direct_inhibition"
    tr = tr_all[tr_all[endpoint].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[endpoint].to_numpy(dtype=float)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    pharma = cyp2c9_feature_matrix(mols)
    X_pharma = np.hstack([X_base, pharma])
    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    repeated_test(["baseline", "+pharma(2)"], [X_base, X_pharma], y, tr, endpoint, base_clusters)


if __name__ == "__main__":
    main()
