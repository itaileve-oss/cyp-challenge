"""CYP3A4-only single-task LightGBM model (fp + 2D descriptors + heme/Ser119
pharmacophore distance), saved as an OOF predictions CSV in the standard format
(other isoform columns left NaN) so it can be blended into the final ensemble.

Why single-task instead of the shared multitask stack: cv_multitask_pharmacophore.py
and cv_multitask_pharmacophore_gated.py both showed the pharmacophore feature makes
the *shared* multitask LightGBM model worse (0.7677 / 0.7684 macro ST-RAE vs 0.7605
baseline), even when explicitly zeroed out for non-CYP3A4 rows -- so it isn't a
leakage/isoform-mismatch problem, it's that one joint tree ensemble across all 4
isoforms' stacked rows can't cleanly exploit a feature that's only real for a
~2335-row subset. Verified directly: in an isolated CYP3A4-only single-task model,
the same feature (as a continuous distance, no need for the binary match flag --
that thresholded version carries no extra signal) genuinely helps:
ST-RAE 0.5416 (fp+desc baseline) -> 0.5347 (+ pharmacophore_best_dist). This script
produces that improved CYP3A4-only model as a standalone ensemble component.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import useful_rdkit_utils as uru
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_multitask import CYP_ISOFORMS, N_FOLDS, SEED  # noqa: E402
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from features_pharmacophore import pharmacophore_feature_matrix  # noqa: E402

FIT_CONFORMERS_CACHE = PROJECT_ROOT / "models" / "fit_conformers_train_cache.pkl"
ENDPOINT = "CYP3A4_pIC50_direct_inhibition"


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    with open(FIT_CONFORMERS_CACHE, "rb") as f:
        fit_mols = pickle.load(f)
    pharma = pharmacophore_feature_matrix(fit_mols)
    best_dist = pharma[:, 2:3]  # continuous distance only -- the binary match flag adds nothing

    X = np.hstack([fp, desc, best_dist])
    y = tr[ENDPOINT].values
    mask = ~np.isnan(y)

    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof = np.full(len(tr), np.nan)
    Xm, ym, cm = X[mask], y[mask], clusters[mask]
    idx_full = np.where(mask)[0]
    for tr_idx, te_idx in gkf.split(Xm, groups=cm):
        model = LGBMRegressor(random_state=SEED, verbosity=-1)
        model.fit(Xm[tr_idx], ym[tr_idx])
        oof[idx_full[te_idx]] = model.predict(Xm[te_idx])

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for isoform in CYP_ISOFORMS:
        col = f"{isoform}_pIC50_direct_inhibition"
        predictions[col] = oof if isoform == "CYP3A4" else np.nan

    out_path = PROJECT_ROOT / "models" / "cv_cyp3a4_pharmacophore_singletask_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Saved OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
