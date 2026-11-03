"""Heme-donor "accessibility" feature: is a candidate heteroatom (N/O/S, same
element set used throughout the docking analysis) positioned near a terminus of
the molecule, or buried mid-scaffold?

Purely 2D/topological -- no conformer, no docking, no QM. Uses RDKit's bond-graph
topological distance matrix directly: an atom's eccentricity (max topological
distance to any other atom in the molecule) normalized by the molecule's diameter
(max eccentricity over all atoms) is close to 1 for an atom near a "head/tail" of
the molecule and close to 0.5 (the graph radius) for one buried near the center.

Motivation: a molecule can carry a chemically plausible heme-ligating group (an
electron-rich N donor) that is nonetheless too sterically buried to physically
reach the iron -- a real, literature-recognized CYP inhibition determinant,
distinct from every earlier hand-crafted 3D pharmacophore/electrostatic feature
tried on this project (all computed on a free-ligand conformer, all either
redundant with existing RDKit descriptors or debunked under repeated-CV).
"""

import numpy as np
from rdkit import Chem

HETEROATOM_ELEMENTS = {"N", "O", "S"}


def accessibility_features_one(smiles: str) -> tuple[float, float, float]:
    """Returns (max_accessibility, mean_accessibility, n_accessible_frac).

    max_accessibility: normalized eccentricity of the best-positioned (most
    terminal) N/O/S candidate -- the primary signal.
    mean_accessibility: mean over all N/O/S candidates.
    n_accessible_frac: fraction of N/O/S candidates with normalized
    eccentricity > 0.7 (an explicit "how many accessible donors" count,
    normalized by heteroatom count so it doesn't just track molecule size).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0, 0.0, 0.0
    mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return 0.0, 0.0, 0.0

    het_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in HETEROATOM_ELEMENTS]
    if not het_idx:
        return 0.0, 0.0, 0.0

    dmat = Chem.GetDistanceMatrix(mol)  # (n_atoms, n_atoms) topological (bond) distances
    diameter = dmat.max()
    if diameter <= 0:
        return 0.0, 0.0, 0.0

    eccentricities = dmat.max(axis=1)  # per-atom eccentricity
    het_acc = eccentricities[het_idx] / diameter

    max_acc = float(het_acc.max())
    mean_acc = float(het_acc.mean())
    frac_accessible = float((het_acc > 0.7).mean())
    return max_acc, mean_acc, frac_accessible


def accessibility_feature_matrix(smiles_list: list[str]) -> np.ndarray:
    rows = [accessibility_features_one(smi) for smi in smiles_list]
    return np.array(rows, dtype=float)
