"""GNN multitask CV with an auxiliary single-concentration-screen output head
(see model_gnn.CypGNNWithAuxiliary and features_single_conc.py for the full
rationale: single-conc data covers ~2500-3000 MORE compounds per isoform than
direct pIC50 labels, correlates strongly with true pIC50 (r=-0.70 to -0.89),
and -- unlike a naive feature-engineering attempt that was tested and made
things worse -- co-training it as a second, lower-weighted loss head lets
those extra compounds shape the shared encoder via gradient.

Same 5-fold scaffold-split CV harness as cv_gnn.py, same architecture
hyperparameters as the winning config from tune_gnn_hyperparams.py (update
GNN_HPARAMS below once that confirms). Only the primary (pIC50) head's
predictions are scored/saved -- the auxiliary head is discarded after
training.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import useful_rdkit_utils as uru
from sklearn.model_selection import GroupKFold, train_test_split
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

from cv_multitask import CYP_ISOFORMS  # noqa: E402
from features_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, mols_to_graphs  # noqa: E402
from features_single_conc import build_log2fc_matrix  # noqa: E402
from model_gnn import CypGNNWithAuxiliary, masked_mse_loss  # noqa: E402

N_FOLDS = 5
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defaults for now -- update to the winning config from tune_gnn_hyperparams.py
# once that confirms (see project memory / tmp/tune_gnn_hyperparams.log).
GNN_HPARAMS = dict(hidden_dim=128, num_layers=4, dropout=0.2, weight_decay=1e-5, lr=1e-3)
AUX_LOSS_WEIGHT = 0.3


def build_labeled_graphs_with_aux(graphs, label_matrix: np.ndarray, aux_matrix: np.ndarray) -> list:
    for g, row, aux_row in zip(graphs, label_matrix, aux_matrix):
        mask = ~np.isnan(row)
        y = np.where(mask, row, 0.0)
        g.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        g.mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

        aux_mask = ~np.isnan(aux_row)
        aux_y = np.where(aux_mask, aux_row, 0.0)
        g.aux_y = torch.tensor(aux_y, dtype=torch.float32).unsqueeze(0)
        g.aux_mask = torch.tensor(aux_mask, dtype=torch.bool).unsqueeze(0)
    return graphs


def train_one_fold(train_graphs, val_graphs):
    model = CypGNNWithAuxiliary(
        ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
        hidden_dim=GNN_HPARAMS["hidden_dim"], num_layers=GNN_HPARAMS["num_layers"], dropout=GNN_HPARAMS["dropout"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_HPARAMS["lr"], weight_decay=GNN_HPARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=256, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred, aux_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            primary_loss = masked_mse_loss(pred, batch.y, batch.mask)
            aux_loss = masked_mse_loss(aux_pred, batch.aux_y, batch.aux_mask)
            loss = primary_loss + AUX_LOSS_WEIGHT * aux_loss
            loss.backward()
            optimizer.step()

        # validate on the PRIMARY task only -- that's what we actually care about
        model.eval()
        val_loss_total, val_mask_total = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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
                print(f"    epoch {epoch}: val_loss={val_loss:.4f} (early stop, best={best_val_loss:.4f})", flush=True)
                break

        if epoch % 5 == 0 or epoch == MAX_EPOCHS - 1:
            print(
                f"    epoch {epoch}: val_loss={val_loss:.4f} best={best_val_loss:.4f} "
                f"({time.time() - epoch_start:.1f}s/epoch)",
                flush=True,
            )

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, graphs) -> np.ndarray:
    model.eval()
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def main() -> None:
    print(f"Using device: {DEVICE}", flush=True)
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()
    mol_names = tr["Molecule_Name"].tolist()

    label_cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    label_matrix = tr[label_cols].to_numpy(dtype=float)

    print("Loading single-conc auxiliary targets...", flush=True)
    aux_matrix = build_log2fc_matrix(mol_names, data_dir / "single_conc_TRAIN.csv")
    print(f"  aux coverage per isoform: {(~np.isnan(aux_matrix)).sum(axis=0)}")

    print("Building molecular graphs...", flush=True)
    graphs = mols_to_graphs(smiles)
    graphs = build_labeled_graphs_with_aux(graphs, label_matrix, aux_matrix)

    print("Computing global scaffold clusters (molecule-level)...", flush=True)
    clusters = uru.get_bemis_murcko_clusters(smiles)
    gkf = GroupKFold(n_splits=N_FOLDS)
    mol_fold = np.full(len(tr), -1)
    for fold, (_, test_mol) in enumerate(gkf.split(np.zeros(len(tr)), groups=clusters)):
        mol_fold[test_mol] = fold

    oof = np.full((len(tr), len(CYP_ISOFORMS)), np.nan)

    for fold in range(N_FOLDS):
        train_mol_idx = np.flatnonzero(mol_fold != fold)
        test_mol_idx = np.flatnonzero(mol_fold == fold)

        inner_train_idx, inner_val_idx = train_test_split(train_mol_idx, test_size=0.1, random_state=SEED)
        print(
            f"Fold {fold}: train {len(inner_train_idx)}, inner-val {len(inner_val_idx)}, "
            f"test {len(test_mol_idx)}",
            flush=True,
        )

        train_graphs = [graphs[i] for i in inner_train_idx]
        val_graphs = [graphs[i] for i in inner_val_idx]
        test_graphs = [graphs[i] for i in test_mol_idx]

        model = train_one_fold(train_graphs, val_graphs)
        preds = predict(model, test_graphs)
        oof[test_mol_idx, :] = preds

    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"]})
    for i, isoform in enumerate(CYP_ISOFORMS):
        predictions[f"{isoform}_pIC50_direct_inhibition"] = oof[:, i]

    out_path = PROJECT_ROOT / "models" / "cv_gnn_auxiliary_oof_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Saved OOF predictions to {out_path}")

    ground_truth_cols = ["Molecule_Name"] + [
        c for c in tr.columns if any(c.startswith(iso) for iso in CYP_ISOFORMS) and "pIC50_direct_inhibition" in c
    ]
    ground_truth = tr[ground_truth_cols]

    bootstrap_results = score_activity_predictions(predictions, ground_truth, REGRESSION_ENDPOINTS)
    track_results = add_macro_endpoint(bootstrap_results, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track_results)

    metric_cols = [f"{name}_mean" for name, _ in ACTIVITY_METRICS] + [f"{name}_std" for name, _ in ACTIVITY_METRICS]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n=== GNN + single-conc auxiliary head, scaffold-split CV ===")
    print(summary[metric_cols].round(4))


if __name__ == "__main__":
    main()
