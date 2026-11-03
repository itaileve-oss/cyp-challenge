"""GNN hyperparameter search, same motivation and protocol as
tune_lgbm_hyperparams.py: the GNN (model_gnn.CypGNN) has never been tuned
either -- hidden_dim=128, num_layers=4, dropout=0.2, weight_decay=1e-5 were
always just reasonable-looking defaults, never validated for this dataset
size (~1300-2300 labels/isoform). Given tuning LightGBM gave a confirmed real
-0.0156, this checks whether the same kind of capacity/regularization
mismatch exists in the GNN.

Two-stage protocol (lighter than the LightGBM version, given GNN training
cost -- each 5-fold run is ~15-20 min vs LightGBM's seconds):
1. Screen ~5 candidates with ONE run each on the fixed scaffold split (same
   split cv_gnn.py always used, so directly comparable to the known default
   number 0.7638/0.7683).
2. Confirm the best candidate against default over 4 repeats (not 8, to keep
   total runtime reasonable) of repeated randomized-cluster CV with a paired
   t-test.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import useful_rdkit_utils as uru
from scipy import stats
from sklearn.model_selection import GroupKFold, train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)
from cv_multitask import CYP_ISOFORMS  # noqa: E402
from features_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, mols_to_graphs  # noqa: E402
from cv_gnn import build_labeled_graphs  # noqa: E402

N_FOLDS = 5
CONFIRM_REPEATS = 4
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TASKS = 4

CANDIDATES = {
    "default": dict(hidden_dim=128, num_layers=4, dropout=0.2, weight_decay=1e-5, lr=1e-3),
    "small_reg": dict(hidden_dim=96, num_layers=3, dropout=0.3, weight_decay=1e-4, lr=1e-3),
    "strong_reg": dict(hidden_dim=64, num_layers=3, dropout=0.4, weight_decay=1e-3, lr=1e-3),
    "deeper_light_reg": dict(hidden_dim=128, num_layers=5, dropout=0.3, weight_decay=1e-4, lr=1e-3),
    "wide_reg": dict(hidden_dim=192, num_layers=4, dropout=0.35, weight_decay=1e-4, lr=5e-4),
}


class TunableCypGNN(nn.Module):
    def __init__(self, atom_feature_dim, bond_feature_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lin0 = nn.Linear(atom_feature_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINEConv(mlp, edge_dim=bond_feature_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.trunk = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dim, N_TASKS)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.lin0(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = torch.relu(norm(conv(h, edge_index, edge_attr)))
            h = h + self.dropout(h_new)
        pooled = torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=1)
        return self.head(self.trunk(pooled))


def masked_mse_loss(pred, target, mask):
    diff_sq = (pred - target) ** 2
    diff_sq = torch.where(mask, diff_sq, torch.zeros_like(diff_sq))
    return diff_sq.sum() / mask.sum().clamp(min=1)


def train_one_fold(train_graphs, val_graphs, hparams, model_seed=42):
    torch.manual_seed(model_seed)
    model = TunableCypGNN(
        ATOM_FEATURE_DIM, BOND_FEATURE_DIM, hparams["hidden_dim"], hparams["num_layers"], hparams["dropout"]
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = masked_mse_loss(pred, batch.y, batch.mask)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_total, val_mask_total = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                diff_sq = ((pred - batch.y) ** 2) * batch.mask
                val_loss_total += diff_sq.sum().item()
                val_mask_total += batch.mask.sum().item()
        val_loss = val_loss_total / max(val_mask_total, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, graphs):
    model.eval()
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def cluster_fold_map(unique_clusters, fold_seed, n_folds=N_FOLDS):
    rng = np.random.RandomState(fold_seed)
    shuffled = rng.permutation(unique_clusters)
    return {c: i % n_folds for i, c in enumerate(shuffled)}


def score_macro(preds, gt):
    boot = score_activity_predictions(preds, gt, REGRESSION_ENDPOINTS)
    track = add_macro_endpoint(boot, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return summary.loc["MA", "ST-RAE_mean"]


def run_gnn_macro(tr, graphs, mol_fold, hparams, gt, model_seed=42):
    oof = np.full((len(graphs), len(CYP_ISOFORMS)), np.nan)
    for fold in range(N_FOLDS):
        train_mol_idx = np.flatnonzero(mol_fold != fold)
        test_mol_idx = np.flatnonzero(mol_fold == fold)
        inner_train_idx, inner_val_idx = train_test_split(train_mol_idx, test_size=0.1, random_state=model_seed)
        train_graphs = [graphs[i] for i in inner_train_idx]
        val_graphs = [graphs[i] for i in inner_val_idx]
        test_graphs = [graphs[i] for i in test_mol_idx]
        model = train_one_fold(train_graphs, val_graphs, hparams, model_seed)
        oof[test_mol_idx, :] = predict(model, test_graphs)
    preds = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, iso in enumerate(CYP_ISOFORMS):
        preds[f"{iso}_pIC50_direct_inhibition"] = oof[:, i]
    return score_macro(preds, gt), oof


def main() -> None:
    print(f"Device: {DEVICE}", flush=True)
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    label_cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    label_matrix = tr[label_cols].to_numpy(dtype=float)
    print("Building graphs...", flush=True)
    graphs = mols_to_graphs(smiles)
    graphs = build_labeled_graphs(graphs, label_matrix)

    clusters = uru.get_bemis_murcko_clusters(smiles)
    gt_cols = ["Molecule_Name"] + [
        c for c in tr.columns if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    gt = tr[gt_cols]

    # Stage 1: screen on the FIXED scaffold split (matches cv_gnn.py's own split)
    gkf = GroupKFold(n_splits=N_FOLDS)
    fixed_mol_fold = np.full(len(tr), -1)
    for fold, (_, test_mol) in enumerate(gkf.split(np.zeros(len(tr)), groups=clusters)):
        fixed_mol_fold[test_mol] = fold

    print(f"\n=== Stage 1: screening {len(CANDIDATES)} candidates on the fixed split ===", flush=True)
    screen_results = {}
    for name, hparams in CANDIDATES.items():
        t0 = time.time()
        score, _ = run_gnn_macro(tr, graphs, fixed_mol_fold, hparams, gt)
        screen_results[name] = score
        print(f"  {name}: MA-ST-RAE={score:.4f} ({time.time()-t0:.0f}s) hparams={hparams}", flush=True)

    ranked = sorted(screen_results.items(), key=lambda kv: kv[1])
    print("\n=== Screening ranking (best first) ===", flush=True)
    for name, score in ranked:
        print(f"  {name}: {score:.4f} (delta vs default: {score - screen_results['default']:+.4f})", flush=True)

    best_name, _ = ranked[0]
    print(f"\nBest candidate: {best_name}", flush=True)
    if best_name == "default":
        print("Default already best -- no confirmation needed.", flush=True)
        return

    print(f"\n=== Stage 2: confirmatory {CONFIRM_REPEATS}-repeat paired comparison: default vs {best_name} ===", flush=True)
    default_scores, best_scores, deltas = [], [], []
    for fold_seed in range(CONFIRM_REPEATS):
        c2f = cluster_fold_map(np.unique(clusters), fold_seed)
        mol_fold = np.array([c2f[c] for c in clusters])
        d_score, d_oof = run_gnn_macro(tr, graphs, mol_fold, CANDIDATES["default"], gt)
        b_score, b_oof = run_gnn_macro(tr, graphs, mol_fold, CANDIDATES[best_name], gt)
        default_scores.append(d_score)
        best_scores.append(b_score)
        deltas.append(b_score - d_score)
        print(f"  fold_seed={fold_seed}: default={d_score:.4f}  {best_name}={b_score:.4f}  delta={b_score-d_score:+.4f}", flush=True)

    default_scores = np.array(default_scores)
    best_scores = np.array(best_scores)
    deltas = np.array(deltas)
    t, p = stats.ttest_1samp(deltas, 0)
    print(f"\ndefault: mean={default_scores.mean():.4f} std={default_scores.std():.4f}")
    print(f"{best_name}: mean={best_scores.mean():.4f} std={best_scores.std():.4f}")
    print(f"paired delta: mean={deltas.mean():+.4f} std={deltas.std():.4f} t={t:.3f} p={p:.4f}")
    print(f"\nWinning hyperparameters: {CANDIDATES[best_name]}")

    if deltas.mean() < 0 and p < 0.05:
        print("\nConfirmed real improvement -- regenerating full OOF on the original fixed split...", flush=True)
        _, final_oof = run_gnn_macro(tr, graphs, fixed_mol_fold, CANDIDATES[best_name], gt)
        predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
        for i, iso in enumerate(CYP_ISOFORMS):
            predictions[f"{iso}_pIC50_direct_inhibition"] = final_oof[:, i]
        out_path = PROJECT_ROOT / "models" / "cv_gnn_tuned_oof_predictions.csv"
        predictions.to_csv(out_path, index=False)
        print(f"Saved tuned GNN OOF predictions to {out_path}")


if __name__ == "__main__":
    main()
