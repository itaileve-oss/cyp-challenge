"""Quick (single-split, not yet repeated-CV) check: does today's improved
CYP3A4 LightGBM (all11 features: docking+pharma+logD, BEST hyperparams from
hparam_search_cyp3a4.py) still give a real lift when 50/50-ensembled with the
existing plain-topology GNN, compared to ensembling the OLD LightGBM
(fp+desc only, old TUNED hyperparams) with that same GNN?

Reuses the existing GNN OOF predictions (models/cv_gnn_oof_predictions.csv,
already trained on the full multitask set) rather than retraining the GNN --
its CYP3A4 column's predictions are legitimately out-of-fold regardless of
whether the LightGBM fold partition matches exactly, so this is a valid
single-split ensemble comparison even though the two models weren't trained
on identical fold splits. A full repeated-CV confirmation would require
retraining the GNN per repeat (~15-20 min each x 8 = 2-2.5h) -- not done here,
this is the fast first-pass read per this session's screening pattern.
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
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import useful_rdkit_utils as uru  # noqa: E402
from evaluation.config import ACTIVITY_METRICS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from features_logd import logd_feature_matrix  # noqa: E402
from cv_repeated_combined_docking_pharma import build_docking, build_pharma_nan  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
N_FOLDS = 5
TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)
BEST = dict(num_leaves=7, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0,
            colsample_bytree=0.5, learning_rate=0.03, n_estimators=800,
            subsample=0.9, subsample_freq=1)

METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]


def all_metrics(oof: np.ndarray, tr: pd.DataFrame) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], ENDPOINT: oof})
    ground_truth = tr[["Molecule_Name", ENDPOINT, f"{ENDPOINT}_conf_low", f"{ENDPOINT}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [ENDPOINT])
    track = add_macro_endpoint(boot, [ENDPOINT], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[ENDPOINT, f"{m}_mean"]) for m in METRIC_NAMES}


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params: dict) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        model = LGBMRegressor(random_state=42, verbosity=-1, **params)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    dock = build_docking(tr)
    pharma = build_pharma_nan(tr)
    logd = logd_feature_matrix(smiles)
    X_base = np.hstack([fp, desc])
    X_all11 = np.hstack([fp, desc, dock, pharma, logd])

    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))

    print("Training old LightGBM (fp+desc, TUNED)...", flush=True)
    oof_lgbm_old = run_fold_split(X_base, y, base_clusters, TUNED)
    print("Training new LightGBM (all11, BEST)...", flush=True)
    oof_lgbm_new = run_fold_split(X_all11, y, base_clusters, BEST)

    print("Loading existing GNN OOF predictions...", flush=True)
    gnn_oof_df = pd.read_csv(PROJECT_ROOT / "models" / "cv_gnn_oof_predictions.csv")
    gnn_lookup = gnn_oof_df.set_index("Molecule_Name")[ENDPOINT]
    oof_gnn = tr["Molecule_Name"].map(gnn_lookup).to_numpy(dtype=float)
    n_missing = np.isnan(oof_gnn).sum()
    print(f"  GNN OOF missing for {n_missing}/{len(tr)} CYP3A4 compounds", flush=True)

    configs = {
        "LightGBM (old: fp+desc, TUNED)": oof_lgbm_old,
        "LightGBM (new: all11, BEST)": oof_lgbm_new,
        "GNN (existing, plain topology)": oof_gnn,
        "50/50 ensemble (old LGBM + GNN)": 0.5 * oof_lgbm_old + 0.5 * oof_gnn,
        "50/50 ensemble (new LGBM + GNN)": 0.5 * oof_lgbm_new + 0.5 * oof_gnn,
    }

    print(f"\n=== Single-split CYP3A4 comparison ===")
    header = f"{'model':34s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for name, oof in configs.items():
        m = all_metrics(oof, tr)
        row = f"{name:34s}" + "".join(f"{m[k]:>14.4f}" for k in METRIC_NAMES)
        print(row)


if __name__ == "__main__":
    main()
