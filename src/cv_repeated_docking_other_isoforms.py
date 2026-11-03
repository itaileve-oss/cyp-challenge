"""Applies the CYP3A4 lesson (docking coordination features: coord_energy,
coord_fe_dist, coord_frac -- coord_first pose selection, mean-of-stereo-per-
receptor then mean-across-receptors aggregation) to CYP1A2/CYP2C9/CYP2D6,
now that their own training-compound docking is done and re-ranked
(rerank_docking_poses_general.py). For CYP2D6 also tests docking stacked on
top of the already-validated basic-amine pharmacophore. Uses the BEST
hyperparameters found for CYP3A4 (hparam_search_cyp3a4.py) as the tuned
config for all three -- not re-optimized per isoform, but a much better prior
than sklearn defaults given what we learned.

Same repeated randomized-cluster-fold harness as every other comparison this
session.
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

N_FOLDS = 5
N_REPEATS = 8
BEST = dict(num_leaves=7, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
            colsample_bytree=0.5, learning_rate=0.03, n_estimators=800,
            subsample=0.9, subsample_freq=1)
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]

ISOFORM_RECEPTORS = {
    "CYP1A2": ["2hi4"],
    "CYP2C9": ["1r9o", "4nz2", "5a5i", "5k7k", "5w0c"],
    "CYP2D6": ["3qm4", "3tbg", "4wnu", "4xrz", "5tft"],
}


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


def build_docking_isoform(isoform: str, tr: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(PROJECT_ROOT / "models" / f"{isoform.lower()}_gd2heme_reranked_raw.csv")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "structures" / f"{isoform.lower()}_variant_manifest.csv")
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5
    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    agg = best_per_receptor.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "mean"), coord_fe_dist=("coord_fe_dist", "mean"),
        coord_frac=("coord_flag", "mean"))
    lookup = agg.set_index("Molecule_Name")
    cols = ["coord_energy", "coord_fe_dist", "coord_frac"]
    arr = np.array([
        lookup.loc[n, cols].to_numpy(dtype=float) if n in lookup.index else [np.nan] * len(cols)
        for n in tr.Molecule_Name
    ])
    means = np.nanmean(arr, axis=0)
    for c in range(arr.shape[1]):
        arr[np.isnan(arr[:, c]), c] = means[c]
    return arr


def run_isoform(isoform: str, tr_all: pd.DataFrame) -> None:
    endpoint = f"{isoform}_pIC50_direct_inhibition"
    tr = tr_all[tr_all[endpoint].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[endpoint].to_numpy(dtype=float)
    print(f"\n=== {isoform}: {len(tr)} compounds ===", flush=True)

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])

    print(f"  Building docking features...", flush=True)
    dock = build_docking_isoform(isoform, tr)
    X_dock = np.hstack([X_base, dock])

    labels = ["baseline", "+docking (3)"]
    feature_sets = [X_base, X_dock]

    if isoform == "CYP2D6":
        print(f"  Building basic-amine pharmacophore features...", flush=True)
        mols = build_mols(smiles, needs_conf=True)
        pharma = cyp2d6_feature_matrix(mols)
        X_pharma = np.hstack([X_base, pharma])
        X_both = np.hstack([X_base, dock, pharma])
        labels += ["+pharma (2)", "+docking+pharma (5)"]
        feature_sets += [X_pharma, X_both]

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
    print(f"\n  --- {isoform} full metric table, mean +/- std over {N_REPEATS} repeats, BEST hyperparams ---")
    header = f"  {'model':20s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"  {lab:20s}"
        for m in METRIC_NAMES:
            vals = np.array(all_results[lab][m])
            row += f"{vals.mean():>8.4f}±{vals.std():.3f}".rjust(14)
        print(row)

    print()
    for i, lab in enumerate(labels[1:], start=1):
        d = st_rae_results[:, i] - st_rae_results[:, 0]
        t, p = ttest_rel(st_rae_results[:, i], st_rae_results[:, 0])
        print(f"  {isoform} {lab} vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  "
              f"t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


def main() -> None:
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    isoform = sys.argv[1] if len(sys.argv) > 1 else None
    isoforms = [isoform] if isoform else list(ISOFORM_RECEPTORS.keys())
    for iso in isoforms:
        run_isoform(iso, tr_all)


if __name__ == "__main__":
    main()
