"""Do the two validated signals from today (coord_first docking: 6/8, p=0.067;
full Kaur pharmacophore under tuned LightGBM: 6/8, p=0.062) stack when combined,
or are they redundant? Same repeated randomized-cluster-fold harness, tuned
LightGBM (where both signals were strongest), native NaN for the pharmacophore
columns.
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


def build_docking(tr: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(PROJECT_ROOT / "models" / "gd2heme_reranked_raw.csv")
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv")
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


def build_pharma_nan(tr: pd.DataFrame) -> np.ndarray:
    import pickle
    with open(fpk.WINNING_POSES_PKL, "rb") as f:
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

    print("Building docking features...", flush=True)
    dock = build_docking(tr)
    print("Building pharmacophore features (native NaN)...", flush=True)
    pharma = build_pharma_nan(tr)

    X_dock = np.hstack([X_base, dock])
    X_pharma = np.hstack([X_base, pharma])
    X_both = np.hstack([X_base, dock, pharma])

    print("Computing base scaffold clusters...", flush=True)
    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    labels = ["baseline", "docking (3)", "pharma (5)", "both (8)"]
    feature_sets = [X_base, X_dock, X_pharma, X_both]
    # per repeat, per model, per metric
    all_results = {lab: {m: [] for m in METRIC_NAMES} for lab in labels}
    st_rae_results = []  # keep for the paired significance tests below

    rng = np.random.default_rng(123)
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]
        row_strae = []
        for lab, X in zip(labels, feature_sets):
            m = all_metrics(run_fold_split(X, y, relabeled, TUNED), tr)
            for k, v in m.items():
                all_results[lab][k].append(v)
            row_strae.append(m["ST-RAE"])
        print(f"repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row_strae)), flush=True)
        st_rae_results.append(row_strae)

    st_rae_results = np.array(st_rae_results)
    print(f"\n=== Full metric table, mean +/- std over {N_REPEATS} repeats, tuned LightGBM ===")
    header = f"{'model':14s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"{lab:14s}"
        for m in METRIC_NAMES:
            vals = np.array(all_results[lab][m])
            row += f"{vals.mean():>8.4f}±{vals.std():.3f}".rjust(14)
        print(row)

    print()
    for i, lab in enumerate(labels[1:], start=1):
        d = st_rae_results[:, i] - st_rae_results[:, 0]
        t, p = ttest_rel(st_rae_results[:, i], st_rae_results[:, 0])
        print(f"{lab} vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")

    d_both_vs_dock = st_rae_results[:, 3] - st_rae_results[:, 1]
    t, p = ttest_rel(st_rae_results[:, 3], st_rae_results[:, 1])
    print(f"\nboth vs docking-only: delta={d_both_vs_dock.mean():+.4f} std={d_both_vs_dock.std():.4f}  t={t:.3f} p={p:.4f}  same-dir={(d_both_vs_dock<0).sum()}/{N_REPEATS}")
    d_both_vs_pharma = st_rae_results[:, 3] - st_rae_results[:, 2]
    t, p = ttest_rel(st_rae_results[:, 3], st_rae_results[:, 2])
    print(f"both vs pharma-only:  delta={d_both_vs_pharma.mean():+.4f} std={d_both_vs_pharma.std():.4f}  t={t:.3f} p={p:.4f}  same-dir={(d_both_vs_pharma<0).sum()}/{N_REPEATS}")


if __name__ == "__main__":
    main()
