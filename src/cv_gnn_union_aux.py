"""GNN multitask CV with TWO auxiliary co-training heads: the existing single-conc log2fc
head, plus a new external-pIC50 head (ChEMBL + PubChem AID1851, leakage-checked against
TEST_BLINDED.csv -- see build_external_pic50.py).

The key structural difference from cv_gnn_auxiliary.py: this trains over a UNION pool --
our own 4,905 labeled molecules PLUS ~25,600 net-new molecules that appear only in the
external data (no OpenADMET pIC50 label at all). The external-only molecules are never
held out for testing (they have nothing to score) -- they get a reserved fold id of -1
and are included in every fold's training set unconditionally. Only the original 4,905
molecules participate in the scaffold-split scoring.

Same 5-fold scaffold-split CV harness and GNN hyperparameters as cv_gnn.py/
cv_gnn_auxiliary.py otherwise, for direct comparability.
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
from features_external_pic50 import build_external_pic50_matrix, inchikey  # noqa: E402
from model_gnn import CypGNNWithUnionAux, masked_mse_loss  # noqa: E402

N_FOLDS = 5
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GNN_HPARAMS = dict(hidden_dim=128, num_layers=4, dropout=0.2, weight_decay=1e-5, lr=1e-3)
AUX_LOSS_WEIGHT = 0.3          # log2fc head (same weight as cv_gnn_auxiliary.py)
AUX2_LOSS_WEIGHT = 0.3         # external pIC50 head -- same starting weight, untuned

EXTERNAL_ONLY_FOLD = -1


def build_union_pool(tr: pd.DataFrame, external_path: Path):
    """Returns (smiles, mol_names_or_none, label_matrix, aux1_matrix, aux2_matrix,
    is_external_only) for the combined pool: our own molecules first, then net-new
    external-only ones appended.
    """
    our_smiles = tr["SMILES"].tolist()
    our_names = tr["Molecule_Name"].tolist()
    label_cols = [f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS]
    our_labels = tr[label_cols].to_numpy(dtype=float)

    print("Loading single-conc auxiliary targets (our own molecules only)...", flush=True)
    aux1_our = build_log2fc_matrix(our_names, PROJECT_ROOT / "data" / "raw" / "single_conc_TRAIN.csv")

    print("Identifying net-new external-only molecules...", flush=True)
    our_keys = {inchikey(s) for s in our_smiles}
    our_keys.discard(None)
    ext = pd.read_csv(external_path)
    ext_only = ext[~ext["inchi_key"].isin(our_keys)].dropna(subset=["smiles"]).reset_index(drop=True)
    print(f"  {len(ext_only):,} net-new external-only molecules", flush=True)

    all_smiles = our_smiles + ext_only["smiles"].tolist()
    n_our, n_ext = len(our_smiles), len(ext_only)

    label_matrix = np.vstack([our_labels, np.full((n_ext, len(CYP_ISOFORMS)), np.nan)])
    aux1_matrix = np.vstack([aux1_our, np.full((n_ext, len(CYP_ISOFORMS)), np.nan)])

    print("Building external-pIC50 auxiliary target matrix over the full union pool...", flush=True)
    aux2_matrix = build_external_pic50_matrix(all_smiles, external_path)

    is_external_only = np.array([False] * n_our + [True] * n_ext)
    return all_smiles, our_names, label_matrix, aux1_matrix, aux2_matrix, is_external_only


def build_labeled_graphs(graphs, label_matrix, aux1_matrix, aux2_matrix):
    for g, row, aux1_row, aux2_row in zip(graphs, label_matrix, aux1_matrix, aux2_matrix):
        mask = ~np.isnan(row)
        y = np.where(mask, row, 0.0)
        g.y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        g.mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

        aux1_mask = ~np.isnan(aux1_row)
        aux1_y = np.where(aux1_mask, aux1_row, 0.0)
        g.aux_y = torch.tensor(aux1_y, dtype=torch.float32).unsqueeze(0)
        g.aux_mask = torch.tensor(aux1_mask, dtype=torch.bool).unsqueeze(0)

        aux2_mask = ~np.isnan(aux2_row)
        aux2_y = np.where(aux2_mask, aux2_row, 0.0)
        g.aux2_y = torch.tensor(aux2_y, dtype=torch.float32).unsqueeze(0)
        g.aux2_mask = torch.tensor(aux2_mask, dtype=torch.bool).unsqueeze(0)
    return graphs


def train_one_fold(train_graphs, val_graphs, max_epochs=MAX_EPOCHS, patience=PATIENCE, verbose=True):
    model = CypGNNWithUnionAux(
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

    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred, aux_pred, aux2_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            primary_loss = masked_mse_loss(pred, batch.y, batch.mask)
            aux_loss = masked_mse_loss(aux_pred, batch.aux_y, batch.aux_mask)
            aux2_loss = masked_mse_loss(aux2_pred, batch.aux2_y, batch.aux2_mask)
            loss = primary_loss + AUX_LOSS_WEIGHT * aux_loss + AUX2_LOSS_WEIGHT * aux2_loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_total, val_mask_total = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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
            if epochs_since_improvement >= patience:
                if verbose:
                    print(f"    epoch {epoch}: val_loss={val_loss:.4f} (early stop, best={best_val_loss:.4f})", flush=True)
                break

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
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
        pred, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def main() -> None:
    print(f"Using device: {DEVICE}", flush=True)
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    external_path = PROJECT_ROOT / "data" / "external" / "external_pic50_clean.csv"

    all_smiles, our_names, label_matrix, aux1_matrix, aux2_matrix, is_external_only = build_union_pool(
        tr, external_path
    )
    n_our = len(our_names)
    print(f"Union pool: {len(all_smiles):,} molecules ({n_our:,} ours + "
          f"{len(all_smiles) - n_our:,} external-only)", flush=True)

    t0 = time.time()
    print("Building molecular graphs for the full union pool...", flush=True)
    graphs = mols_to_graphs(all_smiles)
    graphs = build_labeled_graphs(graphs, label_matrix, aux1_matrix, aux2_matrix)
    print(f"  done in {time.time()-t0:.1f}s, {len(graphs):,} graphs", flush=True)

    print("Computing scaffold clusters (our own molecules only)...", flush=True)
    our_clusters = uru.get_bemis_murcko_clusters(tr["SMILES"].tolist())
    gkf = GroupKFold(n_splits=N_FOLDS)
    mol_fold = np.full(len(all_smiles), EXTERNAL_ONLY_FOLD)
    for fold, (_, test_idx) in enumerate(gkf.split(np.zeros(n_our), groups=our_clusters)):
        mol_fold[test_idx] = fold  # only the first n_our entries ever get a real fold id

    oof = np.full((n_our, len(CYP_ISOFORMS)), np.nan)

    for fold in range(N_FOLDS):
        train_mol_idx = np.flatnonzero(mol_fold != fold)  # includes all external-only rows
        test_mol_idx = np.flatnonzero(mol_fold == fold)    # only ever our own molecules

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

    out_path = PROJECT_ROOT / "models" / "cv_gnn_union_aux_oof_predictions.csv"
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
    print("\n=== GNN + single-conc + external-pIC50 union-pool auxiliary heads, scaffold-split CV ===")
    print(summary[metric_cols].round(4))


if __name__ == "__main__":
    main()
