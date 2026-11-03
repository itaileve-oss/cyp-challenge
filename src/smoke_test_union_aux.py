"""Quick timing/correctness smoke test for cv_gnn_union_aux.py before committing to the
full 200-epoch x 5-fold run: build the union pool, time graph construction, and time a
few epochs on fold 0 only.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_gnn_union_aux import (  # noqa: E402
    PROJECT_ROOT, N_FOLDS, SEED, EXTERNAL_ONLY_FOLD,
    build_union_pool, build_labeled_graphs, train_one_fold,
)
from features_graph import mols_to_graphs  # noqa: E402

t0 = time.time()
tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
external_path = PROJECT_ROOT / "data" / "external" / "external_pic50_clean.csv"
all_smiles, our_names, label_matrix, aux1_matrix, aux2_matrix, is_external_only = build_union_pool(tr, external_path)
n_our = len(our_names)
print(f"Union pool assembled in {time.time()-t0:.1f}s: {len(all_smiles):,} molecules "
      f"({n_our:,} ours + {len(all_smiles)-n_our:,} external-only)", flush=True)

t0 = time.time()
graphs = mols_to_graphs(all_smiles)
print(f"Graph construction: {time.time()-t0:.1f}s for {len(graphs):,} molecules", flush=True)

graphs = build_labeled_graphs(graphs, label_matrix, aux1_matrix, aux2_matrix)

our_clusters = uru.get_bemis_murcko_clusters(tr["SMILES"].tolist())
gkf = GroupKFold(n_splits=N_FOLDS)
mol_fold = np.full(len(all_smiles), EXTERNAL_ONLY_FOLD)
for fold, (_, test_idx) in enumerate(gkf.split(np.zeros(n_our), groups=our_clusters)):
    mol_fold[test_idx] = fold

fold = 0
train_mol_idx = np.flatnonzero(mol_fold != fold)
test_mol_idx = np.flatnonzero(mol_fold == fold)
inner_train_idx, inner_val_idx = train_test_split(train_mol_idx, test_size=0.1, random_state=SEED)
print(f"Fold 0: train {len(inner_train_idx):,}, inner-val {len(inner_val_idx):,}, test {len(test_mol_idx):,}", flush=True)

train_graphs = [graphs[i] for i in inner_train_idx]
val_graphs = [graphs[i] for i in inner_val_idx]

print("Timing 3 epochs on fold 0's expanded training set...", flush=True)
t0 = time.time()
train_one_fold(train_graphs, val_graphs, max_epochs=3, patience=999, verbose=True)
elapsed = time.time() - t0
print(f"\n3 epochs took {elapsed:.1f}s total ({elapsed/3:.1f}s/epoch)", flush=True)
print(f"Extrapolated: 200 epochs x 5 folds = {elapsed/3*200*5/60:.1f} min "
      f"(likely less due to early stopping, but this is the worst case)", flush=True)
