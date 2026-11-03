"""Real TEST_BLINDED.csv predictions from the CYP2D6-only standalone model
(fp+desc + basic-amine/aromatic-ring-distance pharmacophore), trained on the
FULL CYP2D6 training set (no CV holdout) -- same pattern as
predict_test_lgbm_gnn.py, for the standalone blend component that goes into
build_test_submission.py alongside the base ensemble.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_multitask import SEED  # noqa: E402
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from cv_repeated_other_isoform_pharmacophore import build_mols  # noqa: E402
from features_pharmacophore_other_isoforms import cyp2d6_feature_matrix  # noqa: E402

ENDPOINT = "CYP2D6_pIC50_direct_inhibition"
TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr_full = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    tr = tr_full[tr_full[ENDPOINT].notna()].reset_index(drop=True)
    test = pd.read_csv(data_dir / "TEST_BLINDED.csv")

    print("Featurizing train...", flush=True)
    fp_tr = morgan_fp_matrix(tr["SMILES"].tolist())
    desc_tr = rdkit_descriptor_matrix(tr["SMILES"].tolist())
    mols_tr = build_mols(tr["SMILES"].tolist(), needs_conf=True)
    pharma_tr = cyp2d6_feature_matrix(mols_tr)
    X_tr = np.hstack([fp_tr, desc_tr, pharma_tr])
    y_tr = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing test...", flush=True)
    fp_te = morgan_fp_matrix(test["SMILES"].tolist())
    desc_te = rdkit_descriptor_matrix(test["SMILES"].tolist())
    mols_te = build_mols(test["SMILES"].tolist(), needs_conf=True)
    pharma_te = cyp2d6_feature_matrix(mols_te)
    X_te = np.hstack([fp_te, desc_te, pharma_te])

    print(f"Fitting on {len(X_tr)} CYP2D6 training rows (all data, no holdout)...", flush=True)
    model = LGBMRegressor(random_state=SEED, verbosity=-1, **TUNED)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    out = pd.DataFrame({"Molecule_Name": test["Molecule_Name"], ENDPOINT: preds})
    out_path = PROJECT_ROOT / "models" / "test_predictions_cyp2d6_pharmacophore_standalone.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
