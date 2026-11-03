"""Does mean-across-receptors (copied verbatim from CYP3A4's established-best
aggregation, never re-validated for these isoforms) actually beat
best(min-energy)-across-receptors for CYP2C9/CYP2D6's 5-receptor docking
ensembles? Also switches to TUNED hyperparams instead of BEST, since BEST was
just confirmed to hurt CYP2D6 specifically (0/8, p=0.0038) -- using BEST here
would confound the aggregation question with the hyperparameter-transplant
problem already identified.
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
ISOFORMS = ["CYP2C9", "CYP2D6"]


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


def build_docking_mean(isoform: str, tr: pd.DataFrame) -> np.ndarray:
    """Current approach: best-per-receptor-per-stereo, then MEAN across receptors."""
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


def build_docking_best(isoform: str, tr: pd.DataFrame) -> np.ndarray:
    """Alternative: best-per-receptor-per-stereo, then BEST (min energy) across receptors --
    "does this compound fit at least one receptor conformation well" instead of averaging
    fit across all 5."""
    raw = pd.read_csv(PROJECT_ROOT / "models" / f"{isoform.lower()}_gd2heme_reranked_raw.csv")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "structures" / f"{isoform.lower()}_variant_manifest.csv")
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5
    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    best_overall = best_per_receptor.loc[best_per_receptor.groupby("Molecule_Name").coord_energy.idxmin()]
    agg = best_overall.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "first"), coord_fe_dist=("coord_fe_dist", "first"))
    coord_frac = best_per_receptor.groupby("Molecule_Name", as_index=False)["coord_flag"].max()
    coord_frac = coord_frac.rename(columns={"coord_flag": "coord_frac"})
    agg = agg.merge(coord_frac, on="Molecule_Name", how="left")
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

    dock_mean = build_docking_mean(isoform, tr)
    dock_best = build_docking_best(isoform, tr)
    X_mean = np.hstack([X_base, dock_mean])
    X_best = np.hstack([X_base, dock_best])

    labels = ["baseline", "+docking (mean-across-recep)", "+docking (best-across-recep)"]
    feature_sets = [X_base, X_mean, X_best]

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
            m = all_metrics(run_fold_split(X, y, relabeled, TUNED), tr, endpoint)
            for k, v in m.items():
                all_results[lab][k].append(v)
            row_strae.append(m["ST-RAE"])
        print(f"  repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row_strae)), flush=True)
        st_rae_results.append(row_strae)

    st_rae_results = np.array(st_rae_results)
    print(f"\n  --- {isoform} full metric table, mean +/- std over {N_REPEATS} repeats, TUNED hyperparams ---")
    header = f"  {'model':30s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"  {lab:30s}"
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

    d = st_rae_results[:, 2] - st_rae_results[:, 1]
    t, p = ttest_rel(st_rae_results[:, 2], st_rae_results[:, 1])
    print(f"  {isoform} best-across-recep vs mean-across-recep: delta={d.mean():+.4f} std={d.std():.4f}  "
          f"t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


def main() -> None:
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    for iso in ISOFORMS:
        run_isoform(iso, tr_all)


if __name__ == "__main__":
    main()
