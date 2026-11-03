"""CYP2D6-only single-task LightGBM (fp+desc + docking(3) + conformer-based
basic-amine pharmacophore(2), BEST hyperparams), saved as OOF predictions for
blending -- supersedes cv_cyp2d6_pharmacophore_singletask.py now that CYP2D6's
docking data is validated too (docking+pharma combined: p=0.0137, 7/8, the
strongest CYP2D6 result found, per cv_repeated_cyp2d6_docked_pharma.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import RDLogger
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import useful_rdkit_utils as uru  # noqa: E402

from cv_multitask import CYP_ISOFORMS, N_FOLDS, SEED  # noqa: E402
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from cv_repeated_other_isoform_pharmacophore import build_mols  # noqa: E402
from features_pharmacophore_other_isoforms import cyp2d6_feature_matrix  # noqa: E402
from cv_repeated_docking_other_isoforms import build_docking_isoform  # noqa: E402

ENDPOINT = "CYP2D6_pIC50_direct_inhibition"
BEST = dict(num_leaves=7, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
            colsample_bytree=0.5, learning_rate=0.03, n_estimators=800,
            subsample=0.9, subsample_freq=1)


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr_full = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    tr = tr_full[tr_full[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    dock = build_docking_isoform("CYP2D6", tr)
    mols = build_mols(smiles, needs_conf=True)
    pharma = cyp2d6_feature_matrix(mols)
    X = np.hstack([fp, desc, dock, pharma])

    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof = np.full(len(tr), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=clusters):
        model = LGBMRegressor(random_state=SEED, verbosity=-1, **BEST)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for isoform in CYP_ISOFORMS:
        col = f"{isoform}_pIC50_direct_inhibition"
        predictions[col] = oof if isoform == "CYP2D6" else np.nan

    out_path = PROJECT_ROOT / "models" / "cv_cyp2d6_docking_pharma_singletask_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Saved OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
