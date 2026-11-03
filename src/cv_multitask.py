"""Multitask (shared-model) scaffold-split CV for the Direct Inhibition regression track.

Trains a single shared LightGBM model across all 4 isoforms (isoform as a categorical
feature) instead of 4 independent single-task models, on Morgan FP + cleaned RDKit
descriptors. The tutorial notes multitask beat single-task internally, and sharing
data across isoforms should help most for compounds with only 1-2 isoforms measured.

Splitting is done ONCE at the molecule level (global Bemis-Murcko scaffold clusters),
so a given compound's rows for every isoform always land in the same fold -- unlike
running 4 independent per-isoform splits, which would let the same molecule appear in
a training row for one isoform while held out for another, leaking its identity/FP to
a model that also sees its other-isoform label at inference time.

Scored with the same ported evaluation code as src/cv_baseline.py, so the two are
directly comparable on MA-ST-RAE.
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

from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
N_FOLDS = 5
SEED = 42


def build_stacked_dataset(
    tr: pd.DataFrame, base_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Long-format (molecule, isoform) rows: features + isoform code -> label.

    Returns (X, y, mol_idx, iso_code, isoform_col_position). Only rows with a
    non-null label for that isoform are included.
    """
    n_feat = base_features.shape[1]
    X_parts, y_parts, mol_idx_parts, iso_parts = [], [], [], []
    for iso_code, isoform in enumerate(CYP_ISOFORMS):
        label_col = f"{isoform}_pIC50_direct_inhibition"
        mask = tr[label_col].notna().to_numpy()
        idx = np.flatnonzero(mask)
        X_parts.append(base_features[idx])
        y_parts.append(tr.loc[mask, label_col].to_numpy())
        mol_idx_parts.append(idx)
        iso_parts.append(np.full(idx.shape[0], iso_code))

    X = np.vstack(X_parts)
    iso_code_col = np.concatenate(iso_parts).reshape(-1, 1)
    X = np.hstack([X, iso_code_col])
    y = np.concatenate(y_parts)
    mol_idx = np.concatenate(mol_idx_parts)
    iso_code = np.concatenate(iso_parts)
    return X, y, mol_idx, iso_code, n_feat


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")

    print("Featurizing (Morgan FP + RDKit descriptors)...", flush=True)
    fp = morgan_fp_matrix(tr["SMILES"].tolist())
    desc = rdkit_descriptor_matrix(tr["SMILES"].tolist())
    base_features = np.hstack([fp, desc])
    print(f"  fp: {fp.shape}, descriptors: {desc.shape}, combined: {base_features.shape}")

    X, y, mol_idx, iso_code, n_feat = build_stacked_dataset(tr, base_features)
    iso_col = n_feat  # index of the appended isoform-code column

    print("Computing global scaffold clusters (molecule-level)...", flush=True)
    clusters = uru.get_bemis_murcko_clusters(tr["SMILES"].tolist())
    gkf = GroupKFold(n_splits=N_FOLDS)
    mol_fold = np.full(len(tr), -1)
    for fold, (_, test_mol) in enumerate(gkf.split(np.zeros(len(tr)), groups=clusters)):
        mol_fold[test_mol] = fold

    oof = np.full((len(tr), len(CYP_ISOFORMS)), np.nan)
    row_fold = mol_fold[mol_idx]
    for fold in range(N_FOLDS):
        train_rows = row_fold != fold
        test_rows = row_fold == fold
        print(f"Fold {fold}: train rows {train_rows.sum()}, test rows {test_rows.sum()}", flush=True)

        model = LGBMRegressor(random_state=SEED, verbosity=-1)
        model.fit(
            X[train_rows],
            y[train_rows],
            categorical_feature=[iso_col],
        )
        preds = model.predict(X[test_rows])
        oof[mol_idx[test_rows], iso_code[test_rows]] = preds

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, isoform in enumerate(CYP_ISOFORMS):
        predictions[f"{isoform}_pIC50_direct_inhibition"] = oof[:, i]
        n_scored = np.isfinite(oof[:, i]).sum()
        print(f"  {isoform}: {n_scored} out-of-fold predictions")

    ground_truth_cols = ["Molecule_Name"] + [
        c
        for c in tr.columns
        if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    ground_truth = tr[ground_truth_cols]

    bootstrap_results = score_activity_predictions(predictions, ground_truth, REGRESSION_ENDPOINTS)
    track_results = add_macro_endpoint(bootstrap_results, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track_results)

    metric_cols = [f"{name}_mean" for name, _ in ACTIVITY_METRICS] + [
        f"{name}_std" for name, _ in ACTIVITY_METRICS
    ]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== Multitask scaffold-split CV, scored with real evaluation code ===")
    print(summary[metric_cols].round(4))

    out_path = PROJECT_ROOT / "models" / "cv_multitask_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\nSaved OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
