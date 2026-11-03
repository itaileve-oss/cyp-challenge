"""Multitask CV, same as cv_multitask.py, plus GFN2-xTB semi-empirical quantum
descriptors (HOMO/LUMO/gap, electronegativity/hardness, dipole, energy decomposition
-- see features_xtb.py) stacked onto Morgan FP + 2D RDKit descriptors.

Physics-based (quantum electronic structure) rather than empirical/graph-based --
tests whether frontier-orbital/electrostatic info helps, given CYP450 inhibition is
fundamentally redox/electron-transfer chemistry at the heme iron.
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
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from features_xtb import xtb_descriptor_matrix  # noqa: E402

CACHE_XTB = PROJECT_ROOT / "models" / "features_xtb_train_cache.npy"


def get_xtb_features(smiles_list: list[str]) -> np.ndarray:
    if CACHE_XTB.exists():
        print(f"Loading cached xTB descriptors from {CACHE_XTB}")
        return np.load(CACHE_XTB)
    print("Running GFN2-xTB single-point calcs (parallel, 8 workers)...", flush=True)
    mat = xtb_descriptor_matrix(smiles_list, n_workers=8)
    CACHE_XTB.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_XTB, mat)
    return mat


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    print("Featurizing (Morgan FP + 2D RDKit descriptors)...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc2d = rdkit_descriptor_matrix(smiles)

    xtb_raw = get_xtb_features(smiles)
    failed = np.isnan(xtb_raw).all(axis=1).sum()
    print(f"xTB failed for {failed}/{len(smiles)} molecules (filled with column means)")
    col_means = np.nanmean(xtb_raw, axis=0)
    inds = np.where(np.isnan(xtb_raw))
    xtb_raw[inds] = np.take(col_means, inds[1])
    xtb_feat, _ = uru.clean_descriptors(xtb_raw)

    base_features = np.hstack([fp, desc2d, xtb_feat])
    print(
        f"  fp: {fp.shape}, 2D desc: {desc2d.shape}, xTB desc: {xtb_feat.shape}, "
        f"combined: {base_features.shape}"
    )

    X, y, mol_idx, iso_code, n_feat = build_stacked_dataset(tr, base_features)
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
        model.fit(X[train_rows], y[train_rows], categorical_feature=[iso_col])
        preds = model.predict(X[test_rows])
        oof[mol_idx[test_rows], iso_code[test_rows]] = preds

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, isoform in enumerate(CYP_ISOFORMS):
        predictions[f"{isoform}_pIC50_direct_inhibition"] = oof[:, i]

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
    print("\n=== Multitask + GFN2-xTB quantum descriptors, scored with real evaluation code ===")
    print(summary[metric_cols].round(4))


if __name__ == "__main__":
    main()
