"""The key ablation: does explicitly prioritizing the heme-coordinating pose
(coord_first -- the already-validated feature set, 6/8 repeats improved,
p=0.067) actually matter, versus just using GalaxyDock2's own default
lowest-energy pose (rank1, which ignores coordination entirely)?

Same repeated randomized-cluster-fold harness, same aggregation pattern
(best-of-stereo-per-receptor via idxmin, then mean across the 5 receptors),
just swapping which pose gets selected per receptor: plain lowest-energy
(rank1) vs coordination-prioritized (coord_first). Both use the exact same
3-feature shape (energy, fe_dist, coordinating-flag) so this isolates the
"which pose" question from the "how many features" question already settled
against expansion (hm_e, coordinating-only-receptor-mean).
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


def build_features(tr: pd.DataFrame):
    raw = pd.read_csv(PROJECT_ROOT / "models" / "gd2heme_reranked_raw.csv")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv")
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5
    raw["rank1_flag"] = raw["rank1_fe_dist"] < 3.5

    best_coord = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    best_rank1 = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).rank1_energy.idxmin()]

    coord_agg = best_coord.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "mean"), coord_fe_dist=("coord_fe_dist", "mean"),
        coord_frac=("coord_flag", "mean"))
    rank1_agg = best_rank1.groupby("Molecule_Name", as_index=False).agg(
        rank1_energy=("rank1_energy", "mean"), rank1_fe_dist=("rank1_fe_dist", "mean"),
        rank1_frac=("rank1_flag", "mean"))

    def build(agg_df, cols, names):
        lookup = agg_df.set_index("Molecule_Name")
        arr = np.array([
            lookup.loc[n, cols].to_numpy(dtype=float) if n in lookup.index else [np.nan] * len(cols)
            for n in names
        ])
        means = np.nanmean(arr, axis=0)
        for c in range(arr.shape[1]):
            arr[np.isnan(arr[:, c]), c] = means[c]
        return arr

    names = tr["Molecule_Name"].tolist()
    dock_coord = build(coord_agg, ["coord_energy", "coord_fe_dist", "coord_frac"], names)
    dock_rank1 = build(rank1_agg, ["rank1_energy", "rank1_fe_dist", "rank1_frac"], names)
    return dock_coord, dock_rank1


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])

    dock_coord, dock_rank1 = build_features(tr)
    print(f"Fraction identical (coord_first picked the same pose as rank1): "
          f"{np.isclose(dock_coord[:,0], dock_rank1[:,0]).mean():.3f}")
    X_coord = np.hstack([X_base, dock_coord])
    X_rank1 = np.hstack([X_base, dock_rank1])

    print("Computing base scaffold clusters...", flush=True)
    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    results = []
    rng = np.random.default_rng(123)  # same seed -> same fold reassignments as earlier runs
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]

        s_base = st_rae(run_fold_split(X_base, y, relabeled), tr)
        s_coord = st_rae(run_fold_split(X_coord, y, relabeled), tr)
        s_rank1 = st_rae(run_fold_split(X_rank1, y, relabeled), tr)
        print(f"repeat {rep}: baseline={s_base:.4f}  coord_first={s_coord:.4f}  rank1={s_rank1:.4f}", flush=True)
        results.append((s_base, s_coord, s_rank1))

    results = np.array(results)
    print(f"\n=== Summary over {N_REPEATS} repeats ===")
    print(f"baseline:     mean={results[:,0].mean():.4f}  std={results[:,0].std():.4f}")
    print(f"coord_first:  mean={results[:,1].mean():.4f}  std={results[:,1].std():.4f}")
    print(f"rank1:        mean={results[:,2].mean():.4f}  std={results[:,2].std():.4f}")

    d_coord = results[:, 1] - results[:, 0]
    d_rank1 = results[:, 2] - results[:, 0]
    d_diff = results[:, 1] - results[:, 2]
    t1, p1 = ttest_rel(results[:, 1], results[:, 0])
    t2, p2 = ttest_rel(results[:, 2], results[:, 0])
    t3, p3 = ttest_rel(results[:, 1], results[:, 2])
    print(f"\ncoord_first vs baseline: delta={d_coord.mean():+.4f} std={d_coord.std():.4f}  t={t1:.3f} p={p1:.4f}  same-dir={(d_coord<0).sum()}/{N_REPEATS}")
    print(f"rank1 vs baseline:       delta={d_rank1.mean():+.4f} std={d_rank1.std():.4f}  t={t2:.3f} p={p2:.4f}  same-dir={(d_rank1<0).sum()}/{N_REPEATS}")
    print(f"coord_first vs rank1:    delta={d_diff.mean():+.4f} std={d_diff.std():.4f}  t={t3:.3f} p={p3:.4f}  same-dir={(d_diff<0).sum()}/{N_REPEATS}")


if __name__ == "__main__":
    main()
