"""External-data stacking, take 2 after cv_gnn_union_aux.py's failure (2026-09-15,
macro ST-RAE 0.7485 on a merged ChEMBL+PubChem auxiliary head fed to the weaker
topology-only GNN). Two changes: (1) ChEMBL and PubChem/Veith(AID1851) stay as
SEPARATE per-source "heads" -- here, two separate LightGBM stacking models per
isoform rather than one merged auxiliary target -- to avoid the cross-assay-bias
collapse; (2) the pseudo-label features feed our STRONGER backbone (TUNED LightGBM
fp+desc), not the GNN.

For each isoform: fit a LightGBM on ChEMBL-only external data (fp+desc -> chembl
pIC50) and another on PubChem/Veith-only external data, predict both on our own
train/test compounds, and add those two predictions as extra feature columns to
the normal primary LightGBM model. Compared against the same baseline without
those two columns, via the project's standard repeated (8x) 5-fold Bemis-Murcko
cluster CV with paired t-test on ST-RAE (see cv_repeated_cyp2c9_robust_training.py
for the harness this mirrors).

Descriptor columns are computed ONCE per isoform over the union of [chembl pool,
pubchem pool, our train, our test] smiles so `rdkit_descriptor_matrix`'s NaN/inf
column-dropping is consistent across all four blocks (it would otherwise silently
drop different columns depending on what's in each separate call).
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

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
N_FOLDS = 5
N_REPEATS = 8
TUNED = dict(objective="regression", num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]


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


def build_stack_features(iso: str, train: pd.DataFrame, test: pd.DataFrame, ext: pd.DataFrame):
    """Returns (chembl_pred_train, pubchem_pred_train, chembl_pred_test, pubchem_pred_test)."""
    endpoint = f"{iso}_pIC50_direct_inhibition"

    chembl_col, pubchem_col = f"{iso}_chembl_pic50", f"{iso}_pubchem_pic50"
    chembl_pool = ext[ext[chembl_col].notna()][["smiles", chembl_col]].rename(columns={chembl_col: "y"})
    pubchem_pool = ext[ext[pubchem_col].notna()][["smiles", pubchem_col]].rename(columns={pubchem_col: "y"})

    train_smiles = train["SMILES"].tolist()
    test_smiles = test["SMILES"].tolist()

    all_smiles = chembl_pool["smiles"].tolist() + pubchem_pool["smiles"].tolist() + train_smiles + test_smiles
    n_c, n_p, n_tr, n_te = len(chembl_pool), len(pubchem_pool), len(train_smiles), len(test_smiles)
    print(f"  [{iso}] chembl={n_c} pubchem={n_p} train={n_tr} test={n_te}", flush=True)

    fp = morgan_fp_matrix(all_smiles)
    desc = rdkit_descriptor_matrix(all_smiles)
    X = np.hstack([fp, desc])

    X_c, X_p = X[:n_c], X[n_c:n_c + n_p]
    X_tr, X_te = X[n_c + n_p:n_c + n_p + n_tr], X[n_c + n_p + n_tr:]

    model_c = LGBMRegressor(random_state=42, verbosity=-1, **TUNED)
    model_c.fit(X_c, chembl_pool["y"].to_numpy(dtype=float))
    chembl_pred_tr, chembl_pred_te = model_c.predict(X_tr), model_c.predict(X_te)

    model_p = LGBMRegressor(random_state=42, verbosity=-1, **TUNED)
    model_p.fit(X_p, pubchem_pool["y"].to_numpy(dtype=float))
    pubchem_pred_tr, pubchem_pred_te = model_p.predict(X_tr), model_p.predict(X_te)

    return chembl_pred_tr, pubchem_pred_tr, chembl_pred_te, pubchem_pred_te


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr_all = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    test = pd.read_csv(data_dir / "TEST_BLINDED.csv")
    ext = pd.read_csv(PROJECT_ROOT / "data" / "external" / "external_pic50_per_source.csv")

    overall = {}

    for iso in CYP_ISOFORMS:
        endpoint = f"{iso}_pIC50_direct_inhibition"
        tr = tr_all[tr_all[endpoint].notna()].reset_index(drop=True)
        y = tr[endpoint].to_numpy(dtype=float)
        print(f"\n=== {iso}: {len(tr)} labeled compounds ===", flush=True)

        fp_tr = morgan_fp_matrix(tr["SMILES"].tolist())
        desc_tr = rdkit_descriptor_matrix(tr["SMILES"].tolist())
        X_base = np.hstack([fp_tr, desc_tr])

        chembl_pred_tr, pubchem_pred_tr, chembl_pred_te, pubchem_pred_te = build_stack_features(iso, tr, test, ext)
        X_stack = np.hstack([X_base, chembl_pred_tr.reshape(-1, 1), pubchem_pred_tr.reshape(-1, 1)])

        base_clusters = np.array(uru.get_bemis_murcko_clusters(tr["SMILES"].tolist()))
        unique_clusters = np.unique(base_clusters)
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
        cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

        labels = ["baseline (fp+desc)", "+external stack (chembl+pubchem)"]
        all_results = {lab: {m: [] for m in METRIC_NAMES} for lab in labels}
        st_rae_pairs = []
        first_oof_base, first_oof_stack = None, None

        rng = np.random.default_rng(123)
        for rep in range(N_REPEATS):
            perm = rng.permutation(len(unique_clusters))
            relabeled = perm[cluster_int]

            oof_base = run_fold_split(X_base, y, relabeled, TUNED)
            oof_stack = run_fold_split(X_stack, y, relabeled, TUNED)
            if rep == 0:
                first_oof_base, first_oof_stack = oof_base, oof_stack

            row = []
            for lab, oof in zip(labels, [oof_base, oof_stack]):
                m = all_metrics(oof, tr, endpoint)
                for k, v in m.items():
                    all_results[lab][k].append(v)
                row.append(m["ST-RAE"])
            print(f"  repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row)), flush=True)
            st_rae_pairs.append(row)

        st_rae_pairs = np.array(st_rae_pairs)
        print(f"\n  --- {iso} full metric table, mean +/- std over {N_REPEATS} repeats ---")
        header = f"  {'model':34s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
        print(header)
        for lab in labels:
            row = f"  {lab:34s}"
            for m in METRIC_NAMES:
                vals = np.array(all_results[lab][m])
                row += f"{vals.mean():>8.4f}+/-{vals.std():.3f}".rjust(14)
            print(row)

        d = st_rae_pairs[:, 1] - st_rae_pairs[:, 0]
        t, p = ttest_rel(st_rae_pairs[:, 1], st_rae_pairs[:, 0])
        same_dir = int((d < 0).sum())
        print(f"  {iso} stack vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  t={t:.3f} p={p:.4f}  "
              f"same-dir(improved)={same_dir}/{N_REPEATS}")
        overall[iso] = dict(delta=d.mean(), p=p, same_dir=same_dir,
                             base_strae=st_rae_pairs[:, 0].mean(), stack_strae=st_rae_pairs[:, 1].mean())

        pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], endpoint: first_oof_stack}).to_csv(
            PROJECT_ROOT / "models" / f"cv_{iso.lower()}_external_stack_oof_predictions.csv", index=False)

        final_model = LGBMRegressor(random_state=42, verbosity=-1, **TUNED)
        final_model.fit(X_stack, y)
        X_stack_test = np.hstack([
            morgan_fp_matrix(test["SMILES"].tolist()),
            rdkit_descriptor_matrix(test["SMILES"].tolist()),
            chembl_pred_te.reshape(-1, 1), pubchem_pred_te.reshape(-1, 1),
        ])
        test_preds = final_model.predict(X_stack_test)
        pd.DataFrame({"Molecule_Name": test["Molecule_Name"], endpoint: test_preds}).to_csv(
            PROJECT_ROOT / "models" / f"test_predictions_{iso.lower()}_external_stack.csv", index=False)

    print("\n=== SUMMARY: external-stack vs baseline, ST-RAE (lower=better) ===")
    for iso, r in overall.items():
        verdict = "IMPROVED" if r["same_dir"] >= 6 and r["p"] < 0.05 and r["delta"] < 0 else \
                  "WORSE" if r["same_dir"] <= 2 and r["p"] < 0.05 and r["delta"] > 0 else "NO CLEAR EFFECT"
        print(f"  {iso:8s} base={r['base_strae']:.4f} stack={r['stack_strae']:.4f} "
              f"delta={r['delta']:+.4f} p={r['p']:.4f} same_dir={r['same_dir']}/{N_REPEATS}  -> {verdict}")


if __name__ == "__main__":
    main()
