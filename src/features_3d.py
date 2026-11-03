"""3D conformer-derived descriptors: ETKDG embedding + WHIM/GETAWAY/AUTOCORR3D/shape.

Ligand-only 3D shape -- no receptor, no docking. Describes each molecule's own
preferred low-energy conformer, not how it's bound in any CYP active site.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

_SHAPE_FUNCS = [
    rdMolDescriptors.CalcAsphericity,
    rdMolDescriptors.CalcEccentricity,
    rdMolDescriptors.CalcInertialShapeFactor,
    rdMolDescriptors.CalcNPR1,
    rdMolDescriptors.CalcNPR2,
    rdMolDescriptors.CalcRadiusOfGyration,
    rdMolDescriptors.CalcSpherocityIndex,
    rdMolDescriptors.CalcPMI1,
    rdMolDescriptors.CalcPMI2,
    rdMolDescriptors.CalcPMI3,
]

# WHIM=114, GETAWAY=273, AUTOCORR3D=80, shape=10 -- fixed regardless of molecule size.
N_3D_DESCRIPTORS = 114 + 273 + 80 + len(_SHAPE_FUNCS)


def embed_conformer(smiles: str, seed: int = 42):
    """Best-effort ETKDGv3 embedding + MMFF optimization. None on any failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    try:
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None
    return mol


def descriptor_vector(mol) -> np.ndarray:
    """Fixed-length 3D descriptor vector for an embedded, H-added, charged mol."""
    if mol is None:
        return np.full(N_3D_DESCRIPTORS, np.nan)
    try:
        whim = rdMolDescriptors.CalcWHIM(mol)
        getaway = rdMolDescriptors.CalcGETAWAY(mol)
        autocorr3d = rdMolDescriptors.CalcAUTOCORR3D(mol)
        shape = [f(mol) for f in _SHAPE_FUNCS]
        vec = np.array(list(whim) + list(getaway) + list(autocorr3d) + shape, dtype=float)
        if vec.shape[0] != N_3D_DESCRIPTORS:
            return np.full(N_3D_DESCRIPTORS, np.nan)
        return vec
    except Exception:
        return np.full(N_3D_DESCRIPTORS, np.nan)


def conformer_3d_descriptor_matrix(smiles_list: list[str], seed: int = 42) -> np.ndarray:
    """3D descriptors for a list of SMILES. Failed embeddings become all-NaN rows."""
    out = np.full((len(smiles_list), N_3D_DESCRIPTORS), np.nan)
    for i, smi in enumerate(smiles_list):
        mol = embed_conformer(smi, seed=seed)
        out[i] = descriptor_vector(mol)
    return out
