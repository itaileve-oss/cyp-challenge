"""CYP3A4-only single-task GNN: topology (with the rotatable-bond edge flag) +
optional docked-pose electrostatics (features_docked_electrostatics.py) as extra
per-atom features. Isolated single-task, same reasoning as
cv_cyp3a4_docking_singletask.py -- docked-pose data only exists for CYP3A4, so a
shared multitask model would be perturbed by a feature real for only a subset of
rows.

Runs baseline (topology+rotatable only) and +docked-electrostatics back to back
for a same-session, directly comparable single-split A/B.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import torch
import useful_rdkit_utils as uru
from sklearn.model_selection import GroupKFold, train_test_split
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

from features_graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, mols_to_graphs  # noqa: E402
from model_gnn import CypGNN, masked_mse_loss  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"
N_FOLDS = 5
SEED = 42
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 256
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DOCKED_ELEC_CACHE = PROJECT_ROOT / "models" / "features_docked_electrostatics_train_cache.pkl"


def train_one_fold(train_graphs, val_graphs, atom_feature_dim: int) -> CypGNN:
    model = CypGNN(atom_feature_dim, BOND_FEATURE_DIM, n_tasks=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
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
def predict(model: CypGNN, graphs) -> np.ndarray:
    model.eval()
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    preds = []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def run(smiles: list[str], y: np.ndarray, clusters: np.ndarray, extra_atom_features_list) -> np.ndarray:
    atom_feature_dim = ATOM_FEATURE_DIM
    if extra_atom_features_list is not None:
        atom_feature_dim += extra_atom_features_list[0].shape[1]

    graphs = mols_to_graphs(smiles, extra_atom_features_list=extra_atom_features_list)
    for g, yi in zip(graphs, y):
        g.y = torch.tensor([[yi]], dtype=torch.float32)
        g.mask = torch.tensor([[True]], dtype=torch.bool)

    gkf = GroupKFold(n_splits=N_FOLDS)
    mol_fold = np.full(len(smiles), -1)
    for fold, (_, test_idx) in enumerate(gkf.split(np.zeros(len(smiles)), groups=clusters)):
        mol_fold[test_idx] = fold

    oof = np.full(len(smiles), np.nan)
    for fold in range(N_FOLDS):
        train_idx = np.flatnonzero(mol_fold != fold)
        test_idx = np.flatnonzero(mol_fold == fold)
        inner_train_idx, inner_val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED)

        train_graphs = [graphs[i] for i in inner_train_idx]
        val_graphs = [graphs[i] for i in inner_val_idx]
        test_graphs = [graphs[i] for i in test_idx]

        t0 = time.time()
        model = train_one_fold(train_graphs, val_graphs, atom_feature_dim)
        preds = predict(model, test_graphs)
        oof[test_idx] = preds[:, 0]
        print(f"  fold {fold} done in {time.time()-t0:.0f}s", flush=True)

    return oof


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
    print(f"Using device: {DEVICE}", flush=True)
    tr_full = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr_full[tr_full[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)
    print(f"CYP3A4-only subset: {len(tr)} compounds", flush=True)

    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))

    print("\n=== Baseline (topology + rotatable-bond flag, no electrostatics) ===", flush=True)
    oof_base = run(smiles, y, clusters, extra_atom_features_list=None)
    report(oof_base, tr, "baseline (topology+rotatable)")

    print("\n=== +docked electrostatics ===", flush=True)
    with open(DOCKED_ELEC_CACHE, "rb") as f:
        elec_full = pickle.load(f)
    idx_map = {name: i for i, name in enumerate(tr_full["Molecule_Name"])}
    elec_subset = [elec_full[idx_map[name]] for name in tr["Molecule_Name"]]
    oof_elec = run(smiles, y, clusters, extra_atom_features_list=elec_subset)
    report(oof_elec, tr, "+docked electrostatics")

    out = pd.DataFrame({
        "Molecule_Name": tr["Molecule_Name"],
        f"{ENDPOINT}_baseline": oof_base,
        f"{ENDPOINT}_docked_elec": oof_elec,
    })
    out.to_csv(PROJECT_ROOT / "models" / "cv_cyp3a4_docked_gnn_oof_predictions.csv", index=False)
    print("\nSaved OOF predictions to models/cv_cyp3a4_docked_gnn_oof_predictions.csv")


if __name__ == "__main__":
    main()
