"""Multitask LightGBM CV using ONLY Uni-Mol pretrained embeddings (512-dim) --
no Morgan FP, no RDKit descriptors. Tests whether Uni-Mol carries decorrelated
signal worth ensembling with the existing FP-based model, even though stacking it
directly onto the FP+descriptor feature set made things worse (cv_multitask_unimol.py,
0.7900 vs 0.7605 baseline) -- the same logic that made the LightGBM+GNN ensemble
beat both individual components despite the GNN alone being roughly tied with
LightGBM alone.
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

from cv_multitask import CYP_ISOFORMS, N_FOLDS, SEED, build_stacked_dataset  # noqa: E402
from features_unimol import get_cached_embeddings  # noqa: E402

CACHE_UNIMOL = PROJECT_ROOT / "models" / "features_unimol_train_cache.pkl"


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    unimol_emb = get_cached_embeddings(smiles, CACHE_UNIMOL)
    print(f"  unimol embeddings: {unimol_emb.shape}")

    X, y, mol_idx, iso_code, n_feat = build_stacked_dataset(tr, unimol_emb)
    iso_col = n_feat

    print("Computing global scaffold clusters (molecule-level)...", flush=True)
    clusters = uru.get_bemis_murcko_clusters(smiles)
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

    out_path = PROJECT_ROOT / "models" / "cv_unimol_only_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Saved OOF predictions to {out_path}")

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
    print("\n=== Multitask, Uni-Mol embeddings ONLY, scored with real evaluation code ===")
    print(summary[metric_cols].round(4))


if __name__ == "__main__":
    main()
