"""Repeated randomized-cluster-fold validation of the Kaur/Sevrioukova
pharmacophore feature (nearest_aromatic_dist, nearest_amide_dist,
is_coordinating), this time with BOTH fixes established today:
  1. Native NaN pass-through instead of mean-fill (63% of compounds had no
     usable geometry -- mean-filling that many rows was actively hurting the
     earlier single-split test, confirmed by a direct A/B on this project).
  2. Repeated randomized-cluster-fold CV instead of one single split (the
     docking-feature test looked flat/negative on a single split too, before
     repeated testing revealed a real 6/8-repeat directional lean).

Same harness as cv_repeated_docking_singletask.py / cv_repeated_rank1_vs_coordfirst.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import ttest_rel
from sklearn.model_selection import GroupKFold

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
import features_pharmacophore_kaur as fpk  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
N_FOLDS = 5
N_REPEATS = 8
TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)


def st_rae(oof: np.ndarray, tr: pd.DataFrame) -> float:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [ENDPOINT])
    track = add_macro_endpoint(boot, [ENDPOINT], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return float(summary.loc[ENDPOINT, "ST-RAE_mean"])


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params=None) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        kwargs = dict(random_state=42, verbosity=-1)
        if params:
            kwargs.update(params)
        model = LGBMRegressor(**kwargs)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def build_pharma_nan(tr: pd.DataFrame) -> np.ndarray:
    with open(fpk.WINNING_POSES_PKL, "rb") as f:
        import pickle
        poses = pickle.load(f)

    def parse_fe(p):
        for line in Path(p).read_text().splitlines():
            if line.startswith("HETATM") and line[76:78].strip() == "Fe":
                return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    RPD = {
        "4ny4": fpk.PROJECT_ROOT / "data/receptors/cyp3a4_receptor_prepared.pdb",
        "1w0f": fpk.PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_1w0f_prepared.pdb",
        "1w0g": fpk.PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_1w0g_prepared.pdb",
        "3nxu": fpk.PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_3nxu_prepared.pdb",
        "2j0d": fpk.PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_2j0d_prepared.pdb",
    }
    fe_coords = {r: parse_fe(p) for r, p in RPD.items()}

    rows = []
    for name, smi in zip(tr.Molecule_Name, tr.SMILES):
        entry = poses.get(name)
        fe = fe_coords[entry["receptor"]] if entry is not None else None
        r = fpk._one_compound(name, smi, entry, fe) if entry is not None else fpk._empty_result()
        rows.append(r)
    return np.array(rows, dtype=float)


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])

    print("Building pharmacophore features (native NaN)...", flush=True)
    pharma_nan = build_pharma_nan(tr)
    print(f"NaN count per column: {np.isnan(pharma_nan).sum(axis=0)} / {len(pharma_nan)}", flush=True)
    X_pharma = np.hstack([X_base, pharma_nan])

    print("Computing base scaffold clusters...", flush=True)
    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    for param_label, params in [("default", None), ("tuned", TUNED)]:
        print(f"\n=== {param_label} LightGBM ===", flush=True)
        results = []
        rng = np.random.default_rng(123)
        for rep in range(N_REPEATS):
            perm = rng.permutation(len(unique_clusters))
            relabeled = perm[cluster_int]
            s_base = st_rae(run_fold_split(X_base, y, relabeled, params), tr)
            s_pharma = st_rae(run_fold_split(X_pharma, y, relabeled, params), tr)
            delta = s_pharma - s_base
            print(f"repeat {rep}: baseline={s_base:.4f}  +pharma={s_pharma:.4f}  delta={delta:+.4f}", flush=True)
            results.append((s_base, s_pharma))

        results = np.array(results)
        d = results[:, 1] - results[:, 0]
        t, p = ttest_rel(results[:, 1], results[:, 0])
        print(f"\n[{param_label}] baseline: mean={results[:,0].mean():.4f} std={results[:,0].std():.4f}")
        print(f"[{param_label}] +pharma:  mean={results[:,1].mean():.4f} std={results[:,1].std():.4f}")
        print(f"[{param_label}] delta: mean={d.mean():+.4f} std={d.std():.4f}  t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


if __name__ == "__main__":
    main()
