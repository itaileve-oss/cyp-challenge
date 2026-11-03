"""Real TEST_BLINDED.csv predictions from CYP1A2's confirmed-best standalone
config: fp+desc + docking(3) + 4-point pharmacophore (ring+HBA+hydrophobic+
HBD)(4), TUNED hyperparams. Trained on the FULL CYP1A2 training set.
"""

import pickle
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
from features_pharmacophore_cyp1a2_v3 import cyp1a2_pharmacophore_v3_matrix  # noqa: E402

ENDPOINT = "CYP1A2_pIC50_direct_inhibition"
TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)


def build_test_docking(test: pd.DataFrame) -> np.ndarray:
    agg = pd.read_csv(PROJECT_ROOT / "models" / "test_cyp1a2_docking.csv")
    lookup = agg.set_index("Molecule_Name")
    cols = ["coord_energy", "coord_fe_dist", "coord_frac"]
    arr = np.array([
        lookup.loc[n, cols].to_numpy(dtype=float) if n in lookup.index else [np.nan] * len(cols)
        for n in test.Molecule_Name
    ])
    means = np.nanmean(arr, axis=0)
    for c in range(arr.shape[1]):
        arr[np.isnan(arr[:, c]), c] = means[c]
    return arr


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr_full = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    tr = tr_full[tr_full[ENDPOINT].notna()].reset_index(drop=True)
    test = pd.read_csv(data_dir / "TEST_BLINDED.csv")

    print("Featurizing train...", flush=True)
    fp_tr = morgan_fp_matrix(tr["SMILES"].tolist())
    desc_tr = rdkit_descriptor_matrix(tr["SMILES"].tolist())
    from cv_repeated_docking_other_isoforms import build_docking_isoform
    dock_tr = build_docking_isoform("CYP1A2", tr)
    with open(PROJECT_ROOT / "models" / "cyp1a2_winning_docked_poses.pkl", "rb") as f:
        poses_tr = pickle.load(f)
    pharma_tr = cyp1a2_pharmacophore_v3_matrix(tr["Molecule_Name"].tolist(), tr["SMILES"].tolist(), poses_tr)
    X_tr = np.hstack([fp_tr, desc_tr, dock_tr, pharma_tr])
    y_tr = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing test...", flush=True)
    fp_te = morgan_fp_matrix(test["SMILES"].tolist())
    desc_te = rdkit_descriptor_matrix(test["SMILES"].tolist())
    dock_te = build_test_docking(test)
    with open(PROJECT_ROOT / "models" / "test_cyp1a2_winning_docked_poses.pkl", "rb") as f:
        poses_te = pickle.load(f)
    pharma_te = cyp1a2_pharmacophore_v3_matrix(test["Molecule_Name"].tolist(), test["SMILES"].tolist(), poses_te)
    X_te = np.hstack([fp_te, desc_te, dock_te, pharma_te])

    print(f"X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)
    print(f"Fitting on {len(X_tr)} CYP1A2 training rows (all data, no holdout)...", flush=True)
    model = LGBMRegressor(random_state=SEED, verbosity=-1, **TUNED)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    out = pd.DataFrame({"Molecule_Name": test["Molecule_Name"], ENDPOINT: preds})
    out_path = PROJECT_ROOT / "models" / "test_predictions_cyp1a2_final_standalone.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
