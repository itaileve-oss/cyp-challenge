"""Real TEST_BLINDED.csv predictions from the CYP3A4-only standalone model
(fp+desc + docking(3) + pharma(5) + logD(3), BEST hyperparams), trained on
the FULL CYP3A4 training set (no CV holdout) -- same pattern as
predict_test_cyp2d6_pharmacophore.py, using the newly-docked test compounds
(build_test_cyp3a4_docking_features.py) for the docking/pharmacophore parts.
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
from features_logd import logd_feature_matrix  # noqa: E402
import features_pharmacophore_kaur as fpk  # noqa: E402
from cv_repeated_combined_docking_pharma import build_docking  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
BEST = dict(num_leaves=7, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
            colsample_bytree=0.5, learning_rate=0.03, n_estimators=800,
            subsample=0.9, subsample_freq=1)

CYP3A4_FE_COORDS_PDBS = {
    "4ny4": PROJECT_ROOT / "data/receptors/cyp3a4_receptor_prepared.pdb",
    "1w0f": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_1w0f_prepared.pdb",
    "1w0g": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_1w0g_prepared.pdb",
    "3nxu": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_3nxu_prepared.pdb",
    "2j0d": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_2j0d_prepared.pdb",
}


def parse_fe(p):
    for line in Path(p).read_text().splitlines():
        if line.startswith("HETATM") and line[76:78].strip() == "Fe":
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def build_test_docking(test: pd.DataFrame) -> np.ndarray:
    raw = pd.read_csv(PROJECT_ROOT / "models" / "test_cyp3a4_gd2heme_reranked_raw.csv")
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5
    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    agg = best_per_receptor.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "mean"), coord_fe_dist=("coord_fe_dist", "mean"),
        coord_frac=("coord_flag", "mean"))
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


def build_test_pharma(test: pd.DataFrame) -> np.ndarray:
    with open(PROJECT_ROOT / "models" / "test_cyp3a4_winning_docked_poses.pkl", "rb") as f:
        poses = pickle.load(f)
    fe_coords = {r: parse_fe(p) for r, p in CYP3A4_FE_COORDS_PDBS.items()}
    rows = []
    for name, smi in zip(test.Molecule_Name, test.SMILES):
        entry = poses.get(name)
        fe = fe_coords[entry["receptor"]] if entry is not None else None
        r = fpk._one_compound(name, smi, entry, fe) if entry is not None else fpk._empty_result()
        rows.append(r)
    return np.array(rows, dtype=float)


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr_full = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    tr = tr_full[tr_full[ENDPOINT].notna()].reset_index(drop=True)
    test = pd.read_csv(data_dir / "TEST_BLINDED.csv")

    print("Featurizing train...", flush=True)
    fp_tr = morgan_fp_matrix(tr["SMILES"].tolist())
    desc_tr = rdkit_descriptor_matrix(tr["SMILES"].tolist())
    dock_tr = build_docking(tr)
    # native-NaN pharmacophore features (matches the training convention used throughout
    # today's session, not the mean-filled pharmacophore_feature_matrix() default)
    with open(fpk.WINNING_POSES_PKL, "rb") as f:
        train_poses = pickle.load(f)
    fe_coords = {r: parse_fe(p) for r, p in CYP3A4_FE_COORDS_PDBS.items()}
    pharma_tr_rows = []
    for name, smi in zip(tr.Molecule_Name, tr.SMILES):
        entry = train_poses.get(name)
        fe = fe_coords[entry["receptor"]] if entry is not None else None
        r = fpk._one_compound(name, smi, entry, fe) if entry is not None else fpk._empty_result()
        pharma_tr_rows.append(r)
    pharma_tr = np.array(pharma_tr_rows, dtype=float)
    logd_tr = logd_feature_matrix(tr["SMILES"].tolist())
    X_tr = np.hstack([fp_tr, desc_tr, dock_tr, pharma_tr, logd_tr])
    y_tr = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing test...", flush=True)
    fp_te = morgan_fp_matrix(test["SMILES"].tolist())
    desc_te = rdkit_descriptor_matrix(test["SMILES"].tolist())
    dock_te = build_test_docking(test)
    pharma_te = build_test_pharma(test)
    logd_te = logd_feature_matrix(test["SMILES"].tolist())
    X_te = np.hstack([fp_te, desc_te, dock_te, pharma_te, logd_te])

    print(f"X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)
    print(f"Fitting on {len(X_tr)} CYP3A4 training rows (all data, no holdout)...", flush=True)
    model = LGBMRegressor(random_state=SEED, verbosity=-1, **BEST)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    out = pd.DataFrame({"Molecule_Name": test["Molecule_Name"], ENDPOINT: preds})
    out_path = PROJECT_ROOT / "models" / "test_predictions_cyp3a4_docking_pharma_logd_standalone.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
