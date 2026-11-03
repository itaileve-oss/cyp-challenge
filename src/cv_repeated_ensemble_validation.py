"""Repeated randomized-cluster-fold validation of the ensemble decisions behind
the "current best model" claim: does LightGBM+GNN ensembling really beat both
components, and does adding the Uni-Mol blend really help further?

Motivated by discovering (2026-09-14) that every single-split comparison this
project has made carries a real noise floor (isoform-level std ~0.007-0.01,
macro-level std ~0.003-0.006, from re-running the SAME model/features under
different random scaffold-cluster-to-fold assignments) -- and that at least one
claimed "win" (the CYP3A4 pharmacophore blend, delta -0.0069 on the original
fixed split) turned out to be indistinguishable from noise under repeated CV
(paired delta -0.0003, p=0.88). This script applies the same rigor to the much
bigger claims: LightGBM+GNN 50/50 ensembling, and Uni-Mol blending.

For each of N_REPEATS random cluster-to-fold assignments (same assignment used
for every model, so comparisons are paired), trains: LightGBM (fp+desc)
multitask, GNN (topology) multitask, and refits a LightGBM head on cached
Uni-Mol embeddings -- then computes macro MA-ST-RAE for each individually, for
the 50/50 LightGBM+GNN ensemble, and for the Uni-Mol-blended ensemble (weight
0.1, the previously-found-best weight -- not re-optimized per repeat, since
that would itself need nested CV; we're testing whether the DECISION to blend
at that already-chosen weight holds up, not re-tuning the weight).

Expensive: GNN training is ~15-20 min per 5-fold run, so N_REPEATS is kept
small (5) to keep total runtime under ~2 hours. Run in background.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import useful_rdkit_utils as uru
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

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
from features_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, mols_to_graphs  # noqa: E402
from features_unimol import get_cached_embeddings  # noqa: E402
from model_gnn import CypGNN, masked_mse_loss  # noqa: E402
from cv_gnn import build_labeled_graphs, train_one_fold, predict  # noqa: E402

N_REPEATS = 5
N_FOLDS = 5
MODEL_SEED = 42
CACHE_UNIMOL = PROJECT_ROOT / "models" / "features_unimol_train_cache.pkl"
UNIMOL_WEIGHT = 0.1


def cluster_fold_map(unique_clusters, fold_seed, n_folds=N_FOLDS):
    rng = np.random.RandomState(fold_seed)
    shuffled = rng.permutation(unique_clusters)
    return {c: i % n_folds for i, c in enumerate(shuffled)}


def score_macro(preds, gt):
    boot = score_activity_predictions(preds, gt, REGRESSION_ENDPOINTS)
    track = add_macro_endpoint(boot, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return summary.loc["MA", "ST-RAE_mean"]


def run_lgbm_multitask(tr, base_features, row_fold_by_mol_idx):
    X, y, mol_idx, iso_code, n_feat = build_stacked_dataset(tr, base_features)
    iso_col = n_feat
    row_fold = row_fold_by_mol_idx[mol_idx]
    oof = np.full((len(tr), len(CYP_ISOFORMS)), np.nan)
    for fold in range(N_FOLDS):
        tr_idx = row_fold != fold
        te_idx = row_fold == fold
        model = LGBMRegressor(random_state=MODEL_SEED, verbosity=-1)
        model.fit(X[tr_idx], y[tr_idx], categorical_feature=[iso_col])
        oof[mol_idx[te_idx], iso_code[te_idx]] = model.predict(X[te_idx])
    return oof


def run_gnn_multitask(graphs, mol_fold):
    oof = np.full((len(graphs), len(CYP_ISOFORMS)), np.nan)
    for fold in range(N_FOLDS):
        train_mol_idx = np.flatnonzero(mol_fold != fold)
        test_mol_idx = np.flatnonzero(mol_fold == fold)
        inner_train_idx, inner_val_idx = train_test_split(train_mol_idx, test_size=0.1, random_state=MODEL_SEED)
        train_graphs = [graphs[i] for i in inner_train_idx]
        val_graphs = [graphs[i] for i in inner_val_idx]
        test_graphs = [graphs[i] for i in test_mol_idx]
        model = train_one_fold(train_graphs, val_graphs)
        preds = predict(model, test_graphs)
        oof[test_mol_idx, :] = preds
    return oof


def oof_to_preds_df(tr, oof):
    preds = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        preds[f"{iso}_pIC50_direct_inhibition"] = oof[:, i]
    return preds


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    print("Featurizing (this happens once, reused across all repeats)...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    base_features = np.hstack([fp, desc])

    label_cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    label_matrix = tr[label_cols].to_numpy(dtype=float)
    graphs = mols_to_graphs(smiles)
    graphs = build_labeled_graphs(graphs, label_matrix)

    unimol_emb = get_cached_embeddings(smiles, CACHE_UNIMOL)

    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))

    gt_cols = ["Molecule_Name"] + [
        c for c in tr.columns if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    gt = tr[gt_cols]

    lgbm_scores, gnn_scores, ens_scores, unimol_scores, ens_unimol_scores = [], [], [], [], []

    for repeat in range(N_REPEATS):
        t0 = time.time()
        print(f"\n=== Repeat {repeat} ===", flush=True)
        c2f = cluster_fold_map(np.unique(clusters), fold_seed=repeat)
        mol_fold = np.array([c2f[c] for c in clusters])

        print("  training LightGBM (fp+desc) multitask...", flush=True)
        oof_lgbm = run_lgbm_multitask(tr, base_features, mol_fold)

        print("  training GNN (topology) multitask...", flush=True)
        oof_gnn = run_gnn_multitask(graphs, mol_fold)

        print("  training LightGBM (Uni-Mol embeddings) multitask...", flush=True)
        oof_unimol = run_lgbm_multitask(tr, unimol_emb, mol_fold)

        oof_ens = 0.5 * oof_lgbm + 0.5 * oof_gnn
        oof_ens_unimol = (1 - UNIMOL_WEIGHT) * oof_ens + UNIMOL_WEIGHT * oof_unimol

        s_lgbm = score_macro(oof_to_preds_df(tr, oof_lgbm), gt)
        s_gnn = score_macro(oof_to_preds_df(tr, oof_gnn), gt)
        s_ens = score_macro(oof_to_preds_df(tr, oof_ens), gt)
        s_unimol = score_macro(oof_to_preds_df(tr, oof_unimol), gt)
        s_ens_unimol = score_macro(oof_to_preds_df(tr, oof_ens_unimol), gt)

        lgbm_scores.append(s_lgbm)
        gnn_scores.append(s_gnn)
        ens_scores.append(s_ens)
        unimol_scores.append(s_unimol)
        ens_unimol_scores.append(s_ens_unimol)

        print(
            f"  repeat {repeat} ({time.time()-t0:.0f}s): LightGBM={s_lgbm:.4f} GNN={s_gnn:.4f} "
            f"Ensemble={s_ens:.4f} UniMol={s_unimol:.4f} Ensemble+UniMol={s_ens_unimol:.4f}",
            flush=True,
        )

    lgbm_scores = np.array(lgbm_scores)
    gnn_scores = np.array(gnn_scores)
    ens_scores = np.array(ens_scores)
    unimol_scores = np.array(unimol_scores)
    ens_unimol_scores = np.array(ens_unimol_scores)

    print("\n\n=== SUMMARY (N_REPEATS = {}) ===".format(N_REPEATS))
    for name, arr in [
        ("LightGBM alone", lgbm_scores),
        ("GNN alone", gnn_scores),
        ("50/50 Ensemble", ens_scores),
        ("Uni-Mol alone", unimol_scores),
        ("Ensemble + Uni-Mol blend", ens_unimol_scores),
    ]:
        print(f"{name}: mean={arr.mean():.4f} std={arr.std():.4f}")

    delta_ens_vs_lgbm = ens_scores - lgbm_scores
    delta_ens_vs_gnn = ens_scores - gnn_scores
    delta_unimol_blend = ens_unimol_scores - ens_scores

    for name, deltas in [
        ("Ensemble vs LightGBM alone", delta_ens_vs_lgbm),
        ("Ensemble vs GNN alone", delta_ens_vs_gnn),
        ("Uni-Mol blend vs base Ensemble", delta_unimol_blend),
    ]:
        t, p = stats.ttest_1samp(deltas, 0)
        print(
            f"\n{name}: paired delta mean={deltas.mean():+.4f} std={deltas.std():.4f} "
            f"(per-repeat: {np.round(deltas, 4).tolist()}) t={t:.3f} p={p:.4f}"
        )

    out_path = PROJECT_ROOT / "models" / "repeated_cv_ensemble_validation_results.npz"
    np.savez(
        out_path,
        lgbm_scores=lgbm_scores,
        gnn_scores=gnn_scores,
        ens_scores=ens_scores,
        unimol_scores=unimol_scores,
        ens_unimol_scores=ens_unimol_scores,
    )
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
