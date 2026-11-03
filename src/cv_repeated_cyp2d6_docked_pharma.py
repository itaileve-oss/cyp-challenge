"""Does measuring the CYP2D6 basic-amine pharmacophore on the real docked
pose (basic_n_to_fe_dist_docked, matching the literature's actual 5-7A-from-
Fe claim) beat the free-conformer proxy (basic_n_to_ring_dist) that
features_pharmacophore_other_isoforms.py used before any CYP2D6 docking
existed? Tests both alone and combined with docking(3), same repeated
cluster-fold harness, BEST hyperparams (as used throughout today's CYP2D6
validation).
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
from cv_repeated_other_isoform_pharmacophore import build_mols  # noqa: E402
from features_pharmacophore_other_isoforms import cyp2d6_feature_matrix  # noqa: E402
from features_pharmacophore_cyp2d6_docked import cyp2d6_docked_feature_matrix  # noqa: E402
from cv_repeated_docking_other_isoforms import build_docking_isoform  # noqa: E402
from rerank_docking_poses_general import parse_fe_coord  # noqa: E402

ENDPOINT = "CYP2D6_pIC50_direct_inhibition"
N_FOLDS = 5
N_REPEATS = 8
BEST = dict(num_leaves=7, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
            colsample_bytree=0.5, learning_rate=0.03, n_estimators=800,
            subsample=0.9, subsample_freq=1)
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]

CYP2D6_RECEPTORS = {
    r: PROJECT_ROOT / f"data/receptors/cyp2d6/{r}_prepared.pdb"
    for r in ["3qm4", "3tbg", "4wnu", "4xrz", "5tft"]
}


def all_metrics(oof: np.ndarray, tr: pd.DataFrame) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [ENDPOINT])
    track = add_macro_endpoint(boot, [ENDPOINT], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[ENDPOINT, f"{m}_mean"]) for m in METRIC_NAMES}


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params: dict) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1, **params)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def main() -> None:
    import pickle
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr_all[tr_all[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    names = tr["Molecule_Name"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)
    print(f"CYP2D6: {len(tr)} compounds", flush=True)

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])

    print("Building docking features...", flush=True)
    dock = build_docking_isoform("CYP2D6", tr)

    print("Building conformer-based pharmacophore...", flush=True)
    mols = build_mols(smiles, needs_conf=True)
    pharma_conf = cyp2d6_feature_matrix(mols)

    print("Building docked-pose pharmacophore...", flush=True)
    with open(PROJECT_ROOT / "models" / "cyp2d6_winning_docked_poses.pkl", "rb") as f:
        poses = pickle.load(f)
    fe_coords = {r: parse_fe_coord(p) for r, p in CYP2D6_RECEPTORS.items()}
    pharma_docked = cyp2d6_docked_feature_matrix(names, smiles, poses, fe_coords)

    labels = ["baseline", "+docking(3)", "+pharma_conf(2)", "+pharma_docked(2)",
              "+dock+pharma_conf(5)", "+dock+pharma_docked(5)"]
    feature_sets = [
        X_base,
        np.hstack([X_base, dock]),
        np.hstack([X_base, pharma_conf]),
        np.hstack([X_base, pharma_docked]),
        np.hstack([X_base, dock, pharma_conf]),
        np.hstack([X_base, dock, pharma_docked]),
    ]

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
            m = all_metrics(run_fold_split(X, y, relabeled, BEST), tr)
            for k, v in m.items():
                all_results[lab][k].append(v)
            row_strae.append(m["ST-RAE"])
        print(f"repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row_strae)), flush=True)
        st_rae_results.append(row_strae)

    st_rae_results = np.array(st_rae_results)
    print(f"\n=== Full metric table, mean +/- std over {N_REPEATS} repeats, BEST hyperparams ===")
    header = f"{'model':22s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"{lab:22s}"
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
