"""CYP3A4-only single-task LightGBM model (fp + 2D descriptors + heme-coordination
docking features), same isolation pattern as cv_cyp3a4_pharmacophore_singletask.py
and for the same reason: docking data only exists for CYP3A4 (2334/2335 compounds),
so a shared multitask model would perturb its global tree structure for a feature
that's only real for a CYP3A4-sized row subset -- test standalone here first.

Baseline (fp+desc only, CYP3A4 single-task): ST-RAE 0.5416.
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

from cv_multitask import CYP_ISOFORMS, N_FOLDS, SEED  # noqa: E402
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from features_docking import docking_feature_matrix  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"


def run(use_docking: bool) -> tuple[np.ndarray, pd.DataFrame]:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    if use_docking:
        dock = docking_feature_matrix(tr["Molecule_Name"].tolist())
        X = np.hstack([fp, desc, dock])
    else:
        X = np.hstack([fp, desc])

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

    return oof, tr


def report(oof: np.ndarray, tr: pd.DataFrame, label: str) -> None:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]

    cyp3a4_only = [ENDPOINT]
    bootstrap_results = score_activity_predictions(predictions, ground_truth, cyp3a4_only)
    track_results = add_macro_endpoint(bootstrap_results, cyp3a4_only, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track_results)
    metric_cols = [f"{name}_mean" for name, _ in ACTIVITY_METRICS]
    row = summary.loc[ENDPOINT, metric_cols]
    print(f"[{label}] " + "  ".join(f"{m.replace('_mean','')}={row[m]:.4f}" for m in metric_cols))


def main() -> None:
    print("Running baseline (fp+desc only)...")
    oof_base, tr = run(use_docking=False)
    print("Running +docking (coord_first re-ranked, ensemble-aggregated)...")
    oof_dock, _ = run(use_docking=True)

    report(oof_base, tr, "baseline (fp+desc)")
    report(oof_dock, tr, "+docking (coord_first)")

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for isoform in CYP_ISOFORMS:
        col = f"{isoform}_pIC50_direct_inhibition"
        predictions[col] = oof_dock if isoform == "CYP3A4" else np.nan
    out_path = PROJECT_ROOT / "models" / "cv_cyp3a4_docking_singletask_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\nSaved +docking OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
