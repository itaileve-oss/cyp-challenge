"""Stacked meta-learner ensemble: replace the fixed linear blend weights
(50/50 GNN+LightGBM, +0.1 Uni-Mol) with a small learned model per isoform that
combines the 3 base models' predictions -- a fixed global weight can't capture
isoform-specific blending (e.g. GNN might be more reliable for CYP3A4 but less
for CYP2D6) or nonlinear interactions between the base predictions.

Leakage-safe nested stacking: the 3 base OOF prediction files were each
produced via 5-fold scaffold CV on the SAME fixed molecule-level fold
assignment (Bemis-Murcko clusters -> GroupKFold, as in cv_multitask.py) -- so
every row's base-model prediction is already out-of-fold w.r.t. that base
model's own training. To stack without leakage, the meta-model additionally
must never see a row's own fold when learning how to combine predictions for
that fold: for each of the 5 folds, train the meta-model only on stacked rows
whose molecule fold != this fold, then predict on this fold's rows.

Meta-model: a small-capacity LightGBM (few leaves, few trees, isoform as a
categorical feature) on the 3 base predictions -- input is only 3-4 dims, so
overfitting risk is far lower than the original ~2265-dim feature space this
project's base models train on.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)
from cv_multitask import CYP_ISOFORMS, N_FOLDS, SEED  # noqa: E402

META_HPARAMS = dict(num_leaves=7, n_estimators=80, min_child_samples=30, learning_rate=0.1)


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()
    mol_order = tr["Molecule_Name"]

    gnn = pd.read_csv(models_dir / "cv_gnn_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order].reset_index(drop=True)
    lgbm = pd.read_csv(models_dir / "cv_multitask_tuned_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order].reset_index(drop=True)
    unimol = pd.read_csv(models_dir / "cv_unimol_only_oof_predictions.csv").set_index("Molecule_Name").loc[mol_order].reset_index(drop=True)

    # same deterministic fold assignment every base model was trained/OOF'd under
    clusters = uru.get_bemis_murcko_clusters(smiles)
    gkf = GroupKFold(n_splits=N_FOLDS)
    mol_fold = np.full(len(tr), -1)
    for fold, (_, test_mol) in enumerate(gkf.split(np.zeros(len(tr)), groups=clusters)):
        mol_fold[test_mol] = fold

    cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]

    # build stacked (molecule, isoform) meta-training rows
    rows_X, rows_y, rows_mol, rows_iso, rows_fold = [], [], [], [], []
    for i, iso in enumerate(CYP_ISOFORMS):
        col = cols[i]
        y = tr[col].values
        mask = ~np.isnan(y)
        idx = np.flatnonzero(mask)
        feat = np.column_stack([gnn[col].values[idx], lgbm[col].values[idx], unimol[col].values[idx]])
        rows_X.append(feat)
        rows_y.append(y[idx])
        rows_mol.append(idx)
        rows_iso.append(np.full(idx.shape[0], i))
        rows_fold.append(mol_fold[idx])

    X = np.vstack(rows_X)
    y = np.concatenate(rows_y)
    mol_idx = np.concatenate(rows_mol)
    iso_code = np.concatenate(rows_iso)
    row_fold = np.concatenate(rows_fold)
    X = np.hstack([X, iso_code.reshape(-1, 1)])
    iso_col = X.shape[1] - 1

    oof = np.full((len(tr), len(CYP_ISOFORMS)), np.nan)
    for fold in range(N_FOLDS):
        train_rows = row_fold != fold
        test_rows = row_fold == fold
        meta = LGBMRegressor(random_state=SEED, verbosity=-1, **META_HPARAMS)
        meta.fit(X[train_rows], y[train_rows], categorical_feature=[iso_col])
        preds = meta.predict(X[test_rows])
        oof[mol_idx[test_rows], iso_code[test_rows]] = preds

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        predictions[f"{iso}_pIC50_direct_inhibition"] = oof[:, i]

    out_path = models_dir / "cv_stacked_meta_ensemble_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Saved OOF predictions to {out_path}")

    gt_cols = ["Molecule_Name"] + [
        c for c in tr.columns if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    ground_truth = tr[gt_cols]

    bootstrap_results = score_activity_predictions(predictions, ground_truth, REGRESSION_ENDPOINTS)
    track_results = add_macro_endpoint(bootstrap_results, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track_results)

    metric_cols = [f"{name}_mean" for name, _ in ACTIVITY_METRICS] + [f"{name}_std" for name, _ in ACTIVITY_METRICS]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Stacked meta-learner ensemble (GNN + tuned-LightGBM + Uni-Mol) ===")
    print(summary[metric_cols].round(4))


if __name__ == "__main__":
    main()
