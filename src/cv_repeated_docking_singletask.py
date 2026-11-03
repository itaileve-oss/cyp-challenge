"""Repeated randomized-cluster-fold validation of the raw docking features
(coord_energy/coord_fe_dist/coord_frac) on the CYP3A4-only single-task
LightGBM, following this project's established methodology (see
cv_repeated_ensemble_validation.py / tune_lgbm_hyperparams.py precedent):
GroupKFold's fold assignment depends on cluster-label order, not just the
grouping structure (confirmed directly this session -- only 32% fold-number
agreement between two "equivalent" clusterings), so a SINGLE fixed split is
not sufficient to trust a feature -- shuffle the cluster-to-fold mapping
across repeats and look at the paired, per-repeat delta.
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


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])

    raw = pd.read_csv(PROJECT_ROOT / "models" / "gd2heme_reranked_raw.csv")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv")
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5
    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    per_compound = best_per_receptor.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "mean"), coord_fe_dist=("coord_fe_dist", "mean"),
        coord_frac=("coord_flag", "mean"))
    lookup = per_compound.set_index("Molecule_Name")[["coord_energy", "coord_fe_dist", "coord_frac"]]
    dock = np.array([
        lookup.loc[n].to_numpy() if n in lookup.index else [np.nan, np.nan, np.nan]
        for n in tr.Molecule_Name
    ])
    col_means = np.nanmean(dock, axis=0)
    for c in range(3):
        dock[np.isnan(dock[:, c]), c] = col_means[c]
    X_dock = np.hstack([X_base, dock])

    print("Computing base scaffold clusters...", flush=True)
    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    results = []
    rng = np.random.default_rng(123)
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]  # shuffles cluster LABEL ORDER, same grouping structure

        oof_base = run_fold_split(X_base, y, relabeled)
        oof_dock = run_fold_split(X_dock, y, relabeled)
        s_base = st_rae(oof_base, tr)
        s_dock = st_rae(oof_dock, tr)
        delta = s_dock - s_base
        print(f"repeat {rep}: baseline={s_base:.4f}  +docking={s_dock:.4f}  delta={delta:+.4f}", flush=True)
        results.append((s_base, s_dock, delta))

    results = np.array(results)
    t, p = ttest_rel(results[:, 1], results[:, 0])
    print(f"\n=== Summary over {N_REPEATS} repeats ===")
    print(f"baseline:   mean={results[:,0].mean():.4f}  std={results[:,0].std():.4f}")
    print(f"+docking:   mean={results[:,1].mean():.4f}  std={results[:,1].std():.4f}")
    print(f"paired delta: mean={results[:,2].mean():+.4f}  std={results[:,2].std():.4f}")
    print(f"paired t-test: t={t:.3f}  p={p:.4f}")
    same_direction = (results[:, 2] < 0).sum()
    print(f"repeats where docking improved ST-RAE: {same_direction}/{N_REPEATS}")


if __name__ == "__main__":
    main()
