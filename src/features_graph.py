"""Molecule -> graph (atom/bond feature) conversion for the GNN track.

Standard atom/bond featurization (similar to Chemprop's default scheme): one-hot
element identity, degree, formal charge, H count, hybridization, plus aromatic/
in-ring flags for atoms; one-hot bond type/stereo plus conjugated/in-ring/rotatable
flags for bonds. No 3D information, no electrostatics -- topology only, per the
point of this track: let the network learn a representation instead of
hand-engineering one.

Rotatable-bond flag added 2026-09-22 (user's idea): uses RDKit's standard
Lipinski.RotatableBondSmarts definition (single, acyclic, non-terminal bonds,
excludes amide C-N etc.) so the network gets direct per-edge access to
conformational-flexibility information, distinct from the molecule-level
aggregate count already present in the LightGBM track's RDKit descriptor set.

(Tried appending an RDKit Gasteiger partial charge here directly -- made results
slightly worse, see project memory. Electrostatic/multipole features now live in
features_multipole.py as an optional add-on via mol_to_graph's extra_atom_features
param, matching how features_3d.py/features_xtb.py bolt onto the LightGBM track's
base features.py rather than being baked into it.)
"""

import torch
from rdkit import Chem
from rdkit.Chem import Lipinski
from torch_geometric.data import Data

ATOM_ELEMENTS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
ATOM_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
ATOM_NUM_HS = [0, 1, 2, 3, 4]
ATOM_HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]

BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
]

# +1 slot in each one-hot block for "other"/unlisted category.
ATOM_FEATURE_DIM = (
    len(ATOM_ELEMENTS)
    + 1
    + len(ATOM_DEGREES)
    + 1
    + len(ATOM_FORMAL_CHARGES)
    + 1
    + len(ATOM_NUM_HS)
    + 1
    + len(ATOM_HYBRIDIZATIONS)
    + 1
    + 2  # aromatic, in-ring
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + len(BOND_STEREOS) + 1 + 2 + 1  # conjugated, in-ring, rotatable


def _one_hot(value, choices: list) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_ELEMENTS)
        + _one_hot(atom.GetDegree(), ATOM_DEGREES)
        + _one_hot(atom.GetFormalCharge(), ATOM_FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), ATOM_NUM_HS)
        + _one_hot(atom.GetHybridization(), ATOM_HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def _bond_features(bond: Chem.Bond, is_rotatable: bool) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + _one_hot(bond.GetStereo(), BOND_STEREOS)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing()), float(is_rotatable)]
    )


def mol_to_graph(smiles: str, extra_atom_features=None) -> Data:
    """Convert a SMILES string to a PyG ``Data`` graph (undirected, both bond
    directions listed) with atom features in ``x`` and bond features in
    ``edge_attr``. Raises ``ValueError`` if the SMILES cannot be parsed.

    ``extra_atom_features``, if given, is a (n_heavy_atoms, k) array-like of
    additional per-atom features (e.g. multipole moments from
    features_multipole.py) concatenated onto the base topology features, in
    RDKit atom order (i.e. ``Chem.MolFromSmiles(smiles)`` order, no explicit Hs).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    if extra_atom_features is not None:
        extra = torch.as_tensor(extra_atom_features, dtype=torch.float32)
        if extra.shape[0] != x.shape[0]:
            raise ValueError(
                f"extra_atom_features has {extra.shape[0]} rows, expected {x.shape[0]} "
                f"(one per heavy atom) for SMILES {smiles!r}"
            )
        x = torch.cat([x, extra], dim=1)

    rotatable_bond_pairs = {
        frozenset(match) for match in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)
    }

    edge_index_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        is_rotatable = frozenset((i, j)) in rotatable_bond_pairs
        feat = _bond_features(bond, is_rotatable)
        edge_index_list.append((i, j))
        edge_attr_list.append(feat)
        edge_index_list.append((j, i))
        edge_attr_list.append(feat)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    else:
        # single-atom molecule: no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def mols_to_graphs(smiles_list: list[str], extra_atom_features_list=None) -> list[Data]:
    if extra_atom_features_list is None:
        return [mol_to_graph(s) for s in smiles_list]
    return [
        mol_to_graph(s, extra_atom_features=feat)
        for s, feat in zip(smiles_list, extra_atom_features_list)
    ]
