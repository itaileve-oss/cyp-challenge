"""Generate real TEST_BLINDED.csv predictions from the Uni-Mol-embeddings LightGBM
component, trained on the FULL training set. Run in the GNINA conda env (has
unimol_tools + GPU torch). Companion to predict_test_lgbm_gnn.py (different env).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features_unimol import get_cached_embeddings  # noqa: E402

SEED = 42
CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def build_stacked_dataset(tr: pd.DataFrame, base_features: np.ndarray):
    """Inlined from cv_multitask.py to avoid importing useful_rdkit_utils, which
    isn't installed in the GNINA env this script runs in."""
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
    models_dir = PROJECT_ROOT / "models"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    test = pd.read_csv(data_dir / "TEST_BLINDED.csv")

    train_emb = get_cached_embeddings(
        tr["SMILES"].tolist(), models_dir / "features_unimol_train_cache.pkl"
    )
    test_emb = get_cached_embeddings(
        test["SMILES"].tolist(), models_dir / "features_unimol_test_cache.pkl"
    )

    X, y, mol_idx, iso_code, n_feat = build_stacked_dataset(tr, train_emb)
    iso_col = n_feat

    model = LGBMRegressor(random_state=SEED, verbosity=-1)
    print(f"Fitting Uni-Mol LightGBM on {len(X):,} stacked rows...", flush=True)
    model.fit(X, y, categorical_feature=[iso_col])

    preds = np.full((len(test), len(CYP_ISOFORMS)), np.nan)
    for code, iso in enumerate(CYP_ISOFORMS):
        X_test_iso = np.hstack([test_emb, np.full((len(test), 1), code)])
        preds[:, code] = model.predict(X_test_iso)

    out = pd.DataFrame({"Molecule_Name": test["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        out[f"{iso}_pIC50_direct_inhibition"] = preds[:, i]

    out_path = models_dir / "test_predictions_unimol.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved Uni-Mol test predictions -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
