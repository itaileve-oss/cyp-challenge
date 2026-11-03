"""Hyperparameter search for the multitask LightGBM model (fp+desc), using
repeated randomized-cluster CV (not the single fixed split used everywhere
else historically) so the winner is a real, statistically defensible choice
rather than another noise-chasing exercise.

Motivation: every LightGBM run this whole project has used sklearn's untouched
defaults (n_estimators=100, num_leaves=31, no explicit L1/L2, no subsampling).
That's never been validated as appropriate for ~1300-2300 labeled examples per
isoform against a ~2265-dim mostly-binary feature matrix -- a regime where
under-regularization is a real risk (the model has more than enough capacity
to fit noise). This directly tests whether better regularization/capacity
control gives a real improvement.

Two-stage protocol:
1. Screen ~12 candidate configs (including the default as a control) over 3
   repeats each on the macro MA-ST-RAE (cheap: LightGBM-only, no GNN).
2. Take the best-average candidate and re-run it against the default over 8
   repeats (same protocol used to validate/debunk earlier claims this session)
   with a paired t-test, so the final verdict is trustworthy.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru
from lightgbm import LGBMRegressor
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)
from cv_multitask import CYP_ISOFORMS, build_stacked_dataset  # noqa: E402
from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402

N_FOLDS = 5
SCREEN_REPEATS = 3
CONFIRM_REPEATS = 8
N_JOBS = 4  # leave headroom for the concurrent GNN ensemble-validation job

CANDIDATES = {
    "default": {},
    "more_trees_lower_lr": {"n_estimators": 300, "learning_rate": 0.03},
    "fewer_leaves": {"num_leaves": 15},
    "more_leaves": {"num_leaves": 63},
    "min_child_50": {"min_child_samples": 50},
    "min_child_100": {"min_child_samples": 100},
    "l1_l2_reg": {"reg_alpha": 1.0, "reg_lambda": 1.0},
    "strong_l1_l2": {"reg_alpha": 5.0, "reg_lambda": 5.0},
    "feature_subsample": {"colsample_bytree": 0.5},
    "bagging": {"subsample": 0.7, "subsample_freq": 1},
    "regularized_combo": {
        "num_leaves": 15,
        "min_child_samples": 50,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.7,
        "n_estimators": 300,
        "learning_rate": 0.03,
    },
    "light_combo": {
        "num_leaves": 20,
        "min_child_samples": 30,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "n_estimators": 200,
        "learning_rate": 0.05,
    },
}


def cluster_fold_map(unique_clusters, fold_seed, n_folds=N_FOLDS):
    rng = np.random.RandomState(fold_seed)
    shuffled = rng.permutation(unique_clusters)
    return {c: i % n_folds for i, c in enumerate(shuffled)}


def score_macro(preds, gt):
    boot = score_activity_predictions(preds, gt, REGRESSION_ENDPOINTS)
    track = add_macro_endpoint(boot, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return summary.loc["MA", "ST-RAE_mean"]


def run_multitask_macro(tr, base_features, clusters, fold_seed, hparams, gt, model_seed=42):
    X, y, mol_idx, iso_code, n_feat = build_stacked_dataset(tr, base_features)
    iso_col = n_feat
    c2f = cluster_fold_map(np.unique(clusters), fold_seed)
    mol_fold = np.array([c2f[c] for c in clusters])
    row_fold = mol_fold[mol_idx]
    oof = np.full((len(tr), len(CYP_ISOFORMS)), np.nan)
    for fold in range(N_FOLDS):
        tr_idx = row_fold != fold
        te_idx = row_fold == fold
        model = LGBMRegressor(random_state=model_seed, verbosity=-1, n_jobs=N_JOBS, **hparams)
        model.fit(X[tr_idx], y[tr_idx], categorical_feature=[iso_col])
        oof[mol_idx[te_idx], iso_code[te_idx]] = model.predict(X[te_idx])
    preds = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        preds[f"{iso}_pIC50_direct_inhibition"] = oof[:, i]
    return score_macro(preds, gt)


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    base_features = np.hstack([fp, desc])
    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))

    gt_cols = ["Molecule_Name"] + [
        c for c in tr.columns if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    gt = tr[gt_cols]

    print(f"\n=== Stage 1: screening {len(CANDIDATES)} candidates over {SCREEN_REPEATS} repeats ===", flush=True)
    screen_results = {}
    for name, hparams in CANDIDATES.items():
        t0 = time.time()
        scores = [
            run_multitask_macro(tr, base_features, clusters, fold_seed, hparams, gt)
            for fold_seed in range(SCREEN_REPEATS)
        ]
        scores = np.array(scores)
        screen_results[name] = scores
        print(
            f"  {name}: mean={scores.mean():.4f} std={scores.std():.4f} "
            f"scores={np.round(scores, 4).tolist()} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    default_screen_mean = screen_results["default"].mean()
    ranked = sorted(screen_results.items(), key=lambda kv: kv[1].mean())
    print("\n=== Screening ranking (best first) ===", flush=True)
    for name, scores in ranked:
        delta = scores.mean() - default_screen_mean
        print(f"  {name}: mean={scores.mean():.4f} (delta vs default screen mean: {delta:+.4f})", flush=True)

    best_name, best_scores = ranked[0]
    print(f"\nBest candidate from screening: {best_name}", flush=True)

    if best_name == "default":
        print("Default already best in screening -- no further confirmation needed.", flush=True)
        return

    print(f"\n=== Stage 2: confirmatory {CONFIRM_REPEATS}-repeat paired comparison: default vs {best_name} ===", flush=True)
    default_scores, best_config_scores, deltas = [], [], []
    for fold_seed in range(CONFIRM_REPEATS):
        d = run_multitask_macro(tr, base_features, clusters, fold_seed, CANDIDATES["default"], gt)
        b = run_multitask_macro(tr, base_features, clusters, fold_seed, CANDIDATES[best_name], gt)
        default_scores.append(d)
        best_config_scores.append(b)
        deltas.append(b - d)
        print(f"  fold_seed={fold_seed}: default={d:.4f}  {best_name}={b:.4f}  delta={b-d:+.4f}", flush=True)

    default_scores = np.array(default_scores)
    best_config_scores = np.array(best_config_scores)
    deltas = np.array(deltas)
    t, p = stats.ttest_1samp(deltas, 0)
    print(f"\ndefault: mean={default_scores.mean():.4f} std={default_scores.std():.4f}")
    print(f"{best_name}: mean={best_config_scores.mean():.4f} std={best_config_scores.std():.4f}")
    print(f"paired delta: mean={deltas.mean():+.4f} std={deltas.std():.4f} t={t:.3f} p={p:.4f}")
    print(f"\nWinning hyperparameters: {CANDIDATES[best_name]}")


if __name__ == "__main__":
    main()
