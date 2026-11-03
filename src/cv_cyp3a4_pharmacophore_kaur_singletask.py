"""CYP3A4-only single-task LightGBM: fp+desc baseline vs +Kaur pharmacophore
features (nearest aromatic ring / amide group distance from the heme-
coordinating atom, measured on the real docked pose). Same isolation pattern
as cv_cyp3a4_docking_singletask.py.
"""

import pickle
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

from evaluation.config import ACTIVITY_METRICS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

from cv_multitask import CYP_ISOFORMS, N_FOLDS, SEED  # noqa: E402
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
PHARMA_CACHE = PROJECT_ROOT / "models" / "features_pharmacophore_kaur_cache.pkl"


def run(use_pharma: bool) -> tuple[np.ndarray, pd.DataFrame]:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    if use_pharma:
        with open(PHARMA_CACHE, "rb") as f:
            cached = pickle.load(f)
        cache_lookup = dict(zip(cached["Molecule_Name"], cached["feats"]))
        pharma = np.array([cache_lookup[n] for n in tr["Molecule_Name"]])
        X = np.hstack([fp, desc, pharma])
    else:
        X = np.hstack([fp, desc])

    y = tr[ENDPOINT].to_numpy(dtype=float)
    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof = np.full(len(tr), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=clusters):
        model = LGBMRegressor(random_state=SEED, verbosity=-1)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])

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
    oof_base, tr = run(use_pharma=False)
    print("Running +Kaur pharmacophore (docked-pose aromatic/amide distances)...")
    oof_pharma, _ = run(use_pharma=True)

    report(oof_base, tr, "baseline (fp+desc)")
    report(oof_pharma, tr, "+Kaur pharmacophore")


if __name__ == "__main__":
    main()
