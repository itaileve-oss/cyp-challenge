"""Structural (2D, docking-independent) proxy for type I vs type II CYP450
inhibition mechanism.

Type II inhibitors ligate the heme iron directly through a lone pair on an
aromatic sp2 nitrogen (pyridine/imidazole/triazole -- the classic azole
antifungal pharmacophore) or, less commonly, a basic sp3 amine or thioether
sulfur. Type I inhibitors bind in the active-site pocket without engaging the
iron at all. Our docking-derived `is_coordinating` flag (features_docking.py /
features_pharmacophore_kaur.py) is a *pose-level* proxy for this and is only
defined for the ~40-60% of compounds where some docked pose actually reached
coordinating distance -- true type I compounds are indistinguishable from
"docking failed to find the coordinating pose" (both give NaN/0).

This module instead reads the mechanism directly off the 2D structure: does
the molecule even *contain* a canonical iron-ligating functional group? This
is defined for ~100% of compounds (no docking/pose dependency, no
missingness), so it complements rather than duplicates the noisy pose-level
signal.
"""

import numpy as np
from rdkit import Chem

# aromatic N, no H, exactly 2 heavy-atom connections: pyridine-type lone pair
# pointing out of the ring plane (pyridine N, imidazole/triazole/tetrazole
# "n" position) -- the textbook azole-antifungal iron ligand.
PYRIDINE_TYPE_N = Chem.MolFromSmarts("[n;H0;D2]")
# basic sp3 amine: not part of an amide, not aromatic itself, not aniline-type
# (conjugated into a ring is a much weaker/different ligand)
BASIC_AMINE = Chem.MolFromSmarts("[NX3;H0,H1,H2;!$(N-C(=O));!$(N=*);!a;!$(N-a)]")
# divalent, non-oxidized sulfur (thioether) -- a known weaker Fe ligand
THIOETHER_S = Chem.MolFromSmarts("[#16X2;!$([#16]=O)]")

FEATURE_COLS = ["n_pyridine_type_n", "n_basic_amine", "n_thioether_s", "has_fe_ligand_motif"]


def _one_compound(smiles: str) -> tuple:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (0.0, 0.0, 0.0, 0.0)
    n_pyr = len(mol.GetSubstructMatches(PYRIDINE_TYPE_N))
    n_amine = len(mol.GetSubstructMatches(BASIC_AMINE))
    n_thio = len(mol.GetSubstructMatches(THIOETHER_S))
    has_motif = 1.0 if (n_pyr + n_amine + n_thio) > 0 else 0.0
    return (float(n_pyr), float(n_amine), float(n_thio), has_motif)


def mechanism_type_feature_matrix(smiles_list: list[str]) -> np.ndarray:
    rows = [_one_compound(smi) for smi in smiles_list]
    return np.array(rows, dtype=float)
