"""Generate real TEST_BLINDED.csv predictions from the confirmed-best pipeline's
LightGBM (tuned) and GNN (plain topology) components, trained on the FULL training set
(no CV holdout -- this is for the real submission, not validation, which is already
done via repeated CV). Uni-Mol runs separately (predict_test_unimol.py, different env).

Each component gets a small internal train/val split purely for early stopping (GNN) --
not a leak, the same pattern already used inside every CV fold.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import useful_rdkit_utils as uru
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_multitask import CYP_ISOFORMS, build_stacked_dataset  # noqa: E402
from cv_multitask_tuned import TUNED_HPARAMS  # noqa: E402
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from features_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, mols_to_graphs  # noqa: E402
from cv_gnn import build_labeled_graphs, train_one_fold, predict as gnn_predict  # noqa: E402

SEED = 42


def predict_lgbm(tr: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    print("=== LightGBM (tuned) ===", flush=True)
    print("Featurizing train + test...", flush=True)
    fp_tr = morgan_fp_matrix(tr["SMILES"].tolist())
    desc_tr = rdkit_descriptor_matrix(tr["SMILES"].tolist())
    base_tr = np.hstack([fp_tr, desc_tr])

    fp_te = morgan_fp_matrix(test["SMILES"].tolist())
    desc_te = rdkit_descriptor_matrix(test["SMILES"].tolist())
    base_te = np.hstack([fp_te, desc_te])

    X, y, mol_idx, iso_code, n_feat = build_stacked_dataset(tr, base_tr)
    iso_col = n_feat

    model = LGBMRegressor(random_state=SEED, verbosity=-1, **TUNED_HPARAMS)
    print(f"Fitting on {len(X):,} stacked rows (all training data, no holdout)...", flush=True)
    model.fit(X, y, categorical_feature=[iso_col])

    preds = np.full((len(test), len(CYP_ISOFORMS)), np.nan)
    for code, iso in enumerate(CYP_ISOFORMS):
        X_test_iso = np.hstack([base_te, np.full((len(test), 1), code)])
        preds[:, code] = model.predict(X_test_iso)

    out = pd.DataFrame({"Molecule_Name": test["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        out[f"{iso}_pIC50_direct_inhibition"] = preds[:, i]
    return out


def predict_gnn(tr: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    print("\n=== GNN (plain topology) ===", flush=True)
    label_cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    label_matrix = tr[label_cols].to_numpy(dtype=float)

    print("Building graphs for train + test...", flush=True)
    train_graphs = mols_to_graphs(tr["SMILES"].tolist())
    train_graphs = build_labeled_graphs(train_graphs, label_matrix)
    test_graphs = mols_to_graphs(test["SMILES"].tolist())

    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(len(train_graphs)), test_size=0.1, random_state=SEED
    )
    fit_graphs = [train_graphs[i] for i in inner_train_idx]
    val_graphs = [train_graphs[i] for i in inner_val_idx]

    print(f"Training on {len(fit_graphs):,} molecules (all training data minus a small "
          f"early-stopping holdout)...", flush=True)
    t0 = time.time()
    model = train_one_fold(fit_graphs, val_graphs)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    preds = gnn_predict(model, test_graphs)
    out = pd.DataFrame({"Molecule_Name": test["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        out[f"{iso}_pIC50_direct_inhibition"] = preds[:, i]
    return out


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    test = pd.read_csv(data_dir / "TEST_BLINDED.csv")
    print(f"Train: {len(tr):,} molecules, Test: {len(test):,} molecules", flush=True)

    lgbm_preds = predict_lgbm(tr, test)
    lgbm_preds.to_csv(models_dir / "test_predictions_lgbm_tuned.csv", index=False)
    print(f"Saved LightGBM test predictions -> models/test_predictions_lgbm_tuned.csv", flush=True)

    gnn_preds = predict_gnn(tr, test)
    gnn_preds.to_csv(models_dir / "test_predictions_gnn.csv", index=False)
    print(f"Saved GNN test predictions -> models/test_predictions_gnn.csv", flush=True)


if __name__ == "__main__":
    main()
