"""Repeated randomized-cluster-fold validation of an EXPANDED docking feature
set, building on the confirmed-promising v1 (coord_energy/coord_fe_dist/
coord_frac, mean across 5 receptors -- 6/8 repeats improved, p=0.067).

New this round:
  - coord_hm_e: GalaxyDock2-HEME's own heme-specific energy term for the
    coord_first pose (strongest single univariate correlate found all
    session, r=-0.45, but never fed to the model before), mean across 5
    receptors.
  - "coordinating-only" variants of energy/fe_dist/hm_e: instead of averaging
    across all 5 receptors regardless of whether each one actually reaches
    coordination range, average ONLY over the receptors where that specific
    receptor's pose coordinates (<3.5A) -- closer to "pick the pose that
    forms coordination" rather than diluting with non-coordinating receptors.
    Falls back to the plain (all-receptor) mean when no receptor coordinates.

Tests v1-only vs v1+v2 side by side, same repeated-CV harness as
cv_repeated_docking_singletask.py.
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

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
N_FOLDS = 5
N_REPEATS = 8


def st_rae(oof: np.ndarray, tr: pd.DataFrame) -> float:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [ENDPOINT])
    track = add_macro_endpoint(boot, [ENDPOINT], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return float(summary.loc[ENDPOINT, "ST-RAE_mean"])


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def build_docking_features(tr: pd.DataFrame):
    raw = pd.read_csv(PROJECT_ROOT / "models" / "gd2heme_reranked_raw.csv")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv")
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5

    hm = pd.read_csv(PROJECT_ROOT / "models" / "gd2heme_hm_e_raw.csv")

    # merge on (variant_id, receptor) -- the true unique key. Merging on
    # (Molecule_Name, receptor) instead silently fans out into a many-to-many
    # join for any compound with multiple stereoisomer variants (6,815 of the
    # 18,485 rows had a duplicate Molecule_Name+receptor pair) -- caught this
    # by comparing against the earlier v1-only run and finding the numbers
    # didn't reproduce despite identical folds and identical intended features.
    raw = raw.merge(
        hm[["variant_id", "receptor", "coord_hm_e"]],
        on=["variant_id", "receptor"], how="left",
    )
    assert len(raw) == 18485, f"expected 18485 rows after merge, got {len(raw)}"

    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]

    # v1: plain mean across all 5 receptors
    v1 = best_per_receptor.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "mean"), coord_fe_dist=("coord_fe_dist", "mean"),
        coord_frac=("coord_flag", "mean"), coord_hm_e=("coord_hm_e", "mean"))

    # v2: "coordinating-only" mean -- restrict to receptors where that receptor coordinates
    coordinating_only = best_per_receptor[best_per_receptor.coord_flag]
    v2 = coordinating_only.groupby("Molecule_Name", as_index=False).agg(
        coord_energy_co=("coord_energy", "mean"), coord_fe_dist_co=("coord_fe_dist", "mean"),
        coord_hm_e_co=("coord_hm_e", "mean"))

    merged = v1.merge(v2, on="Molecule_Name", how="left")
    # fallback: where no receptor coordinates, use the plain (all-receptor) mean
    for col, base_col in [("coord_energy_co", "coord_energy"), ("coord_fe_dist_co", "coord_fe_dist"),
                           ("coord_hm_e_co", "coord_hm_e")]:
        merged[col] = merged[col].fillna(merged[base_col])

    lookup = merged.set_index("Molecule_Name")
    cols_orig = ["coord_energy", "coord_fe_dist", "coord_frac"]  # the already-validated 3
    cols_hme = ["coord_hm_e"]
    cols_co = ["coord_energy_co", "coord_fe_dist_co", "coord_hm_e_co"]

    def build(names, cols):
        arr = np.array([
            lookup.loc[n, cols].to_numpy(dtype=float) if n in lookup.index else [np.nan] * len(cols)
            for n in names
        ])
        means = np.nanmean(arr, axis=0)
        for c in range(arr.shape[1]):
            arr[np.isnan(arr[:, c]), c] = means[c]
        return arr

    names = tr["Molecule_Name"].tolist()
    dock_orig = build(names, cols_orig)
    dock_plus_hme = build(names, cols_orig + cols_hme)
    dock_plus_all = build(names, cols_orig + cols_hme + cols_co)
    return dock_orig, dock_plus_hme, dock_plus_all


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])

    dock_orig, dock_plus_hme, dock_plus_all = build_docking_features(tr)
    X_orig = np.hstack([X_base, dock_orig])
    X_hme = np.hstack([X_base, dock_plus_hme])
    X_all = np.hstack([X_base, dock_plus_all])

    print("Computing base scaffold clusters...", flush=True)
    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    results = []
    rng = np.random.default_rng(123)  # same seed as v1 run -> same fold reassignments, directly comparable
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]

        s_base = st_rae(run_fold_split(X_base, y, relabeled), tr)
        s_orig = st_rae(run_fold_split(X_orig, y, relabeled), tr)
        s_hme = st_rae(run_fold_split(X_hme, y, relabeled), tr)
        s_all = st_rae(run_fold_split(X_all, y, relabeled), tr)
        print(f"repeat {rep}: baseline={s_base:.4f}  orig(3)={s_orig:.4f}  +hm_e(4)={s_hme:.4f}  +coord_only(7)={s_all:.4f}", flush=True)
        results.append((s_base, s_orig, s_hme, s_all))

    results = np.array(results)
    print(f"\n=== Summary over {N_REPEATS} repeats ===")
    labels = ["baseline", "orig (3 cols)", "+hm_e (4 cols)", "+coord_only (7 cols)"]
    for i, lab in enumerate(labels):
        print(f"{lab:22s} mean={results[:,i].mean():.4f}  std={results[:,i].std():.4f}")

    print()
    for i, lab in enumerate(labels[1:], start=1):
        d = results[:, i] - results[:, 0]
        t, p = ttest_rel(results[:, i], results[:, 0])
        print(f"{lab} vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


if __name__ == "__main__":
    main()
