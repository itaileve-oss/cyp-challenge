"""Loads and aggregates the single-concentration screen (single_conc_TRAIN.csv)
into a per-molecule, per-isoform auxiliary regression target (log2fc_estimate,
z-scored per isoform) for multi-task co-training.

Why this data is valuable despite not being usable as a direct pIC50 model
input feature (it doesn't exist for the blind test set, only SMILES does):
it covers ALL 4376 single-conc-screened compounds against all 4 enzymes, while
direct dose-response pIC50 labels only cover 1285-2335 compounds per isoform --
roughly 2500-3000 EXTRA compounds per isoform have a log2fc measurement but no
pIC50 label. Correlation with true pIC50 (checked before building this) is
very strong: r=-0.70 to -0.89 across the 4 isoforms, far stronger than any
other signal found this session -- but a naive "predict log2fc from SMILES,
use the prediction as an extra input feature" approach was tested and made
things WORSE (it's derived from the exact same base features the main model
already has, so it can't add information, only auxiliary-model noise). The
correct use is as an auxiliary co-training TARGET on a shared representation
(e.g. an extra GNN output head with masked loss) -- see model_gnn.py's
auxiliary-head class and cv_gnn_auxiliary.py -- so the extra ~2500-3000
compounds per isoform get to shape the shared learned representation via
gradient, which a fixed post-hoc feature cannot do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def build_log2fc_matrix(mol_names: list, single_conc_path: Path) -> np.ndarray:
    """Returns (N, 4) z-scored (per-isoform) log2fc_estimate matrix, NaN where
    a molecule has no single-conc measurement for that isoform.
    """
    sc = pd.read_csv(single_conc_path)
    agg = sc.groupby(["Molecule_Name", "enzyme"])["log2fc_estimate"].mean().unstack("enzyme")

    matrix = np.full((len(mol_names), len(CYP_ISOFORMS)), np.nan)
    mol_to_row = {m: i for i, m in enumerate(mol_names)}
    for iso_idx, iso in enumerate(CYP_ISOFORMS):
        if iso not in agg.columns:
            continue
        col = agg[iso]
        mean, std = col.mean(), col.std()
        z = (col - mean) / std
        for mol_name, val in z.items():
            row = mol_to_row.get(mol_name)
            if row is not None and not np.isnan(val):
                matrix[row, iso_idx] = val

    return matrix
