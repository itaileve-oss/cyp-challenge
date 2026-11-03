"""Repeated randomized-cluster-fold validation of the single-conc auxiliary-head
GNN (cv_gnn_auxiliary.py) against the plain topology-only GNN (cv_gnn.py).

Motivated by: the auxiliary run finished training successfully on 2026-09-15
but its background shell died mid-scoring before printing a final number. The
OOF predictions were rescored from disk on the single original fixed split:
MA-ST-RAE = 0.7511 (aux) vs 0.7638 (plain, fixed split) / 0.7683+/-0.0102
(plain, repeated-CV mean, confirmed 2026-09-14). The gap looks bigger than the
GNN's own measured noise floor, but per this project's established standard
(see project_cyp_challenge memory item 5d), no single-split delta is trusted
without repeated-CV confirmation -- this script does that.

For each of N_REPEATS random cluster-to-fold assignments (same assignment
used for both models each repeat, so the comparison is paired), trains: plain
GNN (topology only, cv_gnn.py's CypGNN) and auxiliary GNN (topology +
single-conc co-training head, cv_gnn_auxiliary.py's CypGNNWithAuxiliary) --
then computes macro MA-ST-RAE for each and the paired delta.

Expensive: two full 5-fold GNN trainings per repeat (~15-20 min each), so
N_REPEATS=5 is expected to take ~2.5-3.5 hours total. Run in background.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import useful_rdkit_utils as uru
from scipy import stats
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

from cv_gnn import CYP_ISOFORMS  # noqa: E402
from cv_gnn import train_one_fold as train_one_fold_plain  # noqa: E402
from cv_gnn import predict as predict_plain  # noqa: E402
from cv_gnn_auxiliary import build_labeled_graphs_with_aux  # noqa: E402
from cv_gnn_auxiliary import train_one_fold as train_one_fold_aux  # noqa: E402
from cv_gnn_auxiliary import predict as predict_aux  # noqa: E402
from features_graph import mols_to_graphs  # noqa: E402
from features_single_conc import build_log2fc_matrix  # noqa: E402

N_REPEATS = 5
N_FOLDS = 5
MODEL_SEED = 42


def cluster_fold_map(unique_clusters, fold_seed, n_folds=N_FOLDS):
    rng = np.random.RandomState(fold_seed)
    shuffled = rng.permutation(unique_clusters)
    return {c: i % n_folds for i, c in enumerate(shuffled)}


def score_macro(preds, gt):
    boot = score_activity_predictions(preds, gt, REGRESSION_ENDPOINTS)
    track = add_macro_endpoint(boot, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return summary.loc["MA", "ST-RAE_mean"]


def oof_to_preds_df(tr, oof):
    preds = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        preds[f"{iso}_pIC50_direct_inhibition"] = oof[:, i]
    return preds


def run_plain(graphs, mol_fold):
    oof = np.full((len(graphs), len(CYP_ISOFORMS)), np.nan)
    for fold in range(N_FOLDS):
        train_mol_idx = np.flatnonzero(mol_fold != fold)
        test_mol_idx = np.flatnonzero(mol_fold == fold)
        inner_train_idx, inner_val_idx = train_test_split(train_mol_idx, test_size=0.1, random_state=MODEL_SEED)
        train_graphs = [graphs[i] for i in inner_train_idx]
        val_graphs = [graphs[i] for i in inner_val_idx]
        test_graphs = [graphs[i] for i in test_mol_idx]
        model = train_one_fold_plain(train_graphs, val_graphs)
        oof[test_mol_idx, :] = predict_plain(model, test_graphs)
    return oof


def run_aux(graphs, mol_fold):
    oof = np.full((len(graphs), len(CYP_ISOFORMS)), np.nan)
    for fold in range(N_FOLDS):
        train_mol_idx = np.flatnonzero(mol_fold != fold)
        test_mol_idx = np.flatnonzero(mol_fold == fold)
        inner_train_idx, inner_val_idx = train_test_split(train_mol_idx, test_size=0.1, random_state=MODEL_SEED)
        train_graphs = [graphs[i] for i in inner_train_idx]
        val_graphs = [graphs[i] for i in inner_val_idx]
        test_graphs = [graphs[i] for i in test_mol_idx]
        model = train_one_fold_aux(train_graphs, val_graphs)
        oof[test_mol_idx, :] = predict_aux(model, test_graphs)
    return oof


def main() -> None:
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()
    mol_names = tr["Molecule_Name"].tolist()

    label_cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    label_matrix = tr[label_cols].to_numpy(dtype=float)

    print("Loading single-conc auxiliary targets...", flush=True)
    aux_matrix = build_log2fc_matrix(mol_names, data_dir / "single_conc_TRAIN.csv")

    print("Building molecular graphs (shared between both models)...", flush=True)
    graphs = mols_to_graphs(smiles)
    graphs = build_labeled_graphs_with_aux(graphs, label_matrix, aux_matrix)

    print("Computing global scaffold clusters (molecule-level)...", flush=True)
    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))

    gt_cols = ["Molecule_Name"] + [
        c for c in tr.columns if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    gt = tr[gt_cols]

    plain_scores, aux_scores = [], []

    for repeat in range(N_REPEATS):
        t0 = time.time()
        print(f"\n=== Repeat {repeat} ===", flush=True)
        c2f = cluster_fold_map(np.unique(clusters), fold_seed=repeat)
        mol_fold = np.array([c2f[c] for c in clusters])

        print("  training plain GNN (topology only)...", flush=True)
        oof_plain = run_plain(graphs, mol_fold)

        print("  training auxiliary GNN (topology + single-conc head)...", flush=True)
        oof_aux = run_aux(graphs, mol_fold)

        s_plain = score_macro(oof_to_preds_df(tr, oof_plain), gt)
        s_aux = score_macro(oof_to_preds_df(tr, oof_aux), gt)

        plain_scores.append(s_plain)
        aux_scores.append(s_aux)

        print(
            f"  repeat {repeat} ({time.time()-t0:.0f}s): Plain={s_plain:.4f} Auxiliary={s_aux:.4f} "
            f"delta={s_aux - s_plain:+.4f}",
            flush=True,
        )

    plain_scores = np.array(plain_scores)
    aux_scores = np.array(aux_scores)

    print("\n\n=== SUMMARY (N_REPEATS = {}) ===".format(N_REPEATS))
    for name, arr in [("Plain GNN", plain_scores), ("Auxiliary GNN", aux_scores)]:
        print(f"{name}: mean={arr.mean():.4f} std={arr.std():.4f}")

    deltas = aux_scores - plain_scores
    t, p = stats.ttest_1samp(deltas, 0)
    print(
        f"\nAuxiliary vs Plain: paired delta mean={deltas.mean():+.4f} std={deltas.std():.4f} "
        f"(per-repeat: {np.round(deltas, 4).tolist()}) t={t:.3f} p={p:.4f}"
    )

    out_path = PROJECT_ROOT / "models" / "repeated_cv_gnn_auxiliary_validation_results.npz"
    np.savez(out_path, plain_scores=plain_scores, aux_scores=aux_scores)
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
