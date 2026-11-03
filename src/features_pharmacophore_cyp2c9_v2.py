"""CYP2C9 pharmacophore v2, per the literature: a lipophilic aromatic ring
paired with an ionized acidic OR hydrogen-bond-acceptor group that
coordinates the heme iron -- explicitly including azole/pyridine RING
nitrogens (fluconazole, ketoconazole -- the coordinating N is part of the
aromatic ring itself) and sulfonamide oxygen (sulfaphenazole, a named
classic CYP2C9 inhibitor), neither of which the earlier acidic-group
features (carboxylic acid/tetrazole/sulfonic acid only) or ring-pharmacophore
(which excluded ring atoms from the acceptor search) could represent.

Two docked-pose distances:
  - coord_atom_to_fe_dist: nearest "coordinating-capable" atom (pyridine-type
    ring N, carboxylic/tetrazole/sulfonic acidic O, or sulfonamide O) to Fe
  - ring_to_coord_dist: distance from the largest lipophilic aromatic ring's
    centroid to that same atom (can be ~0 if the coordinating N IS the ring,
    which is itself informative -- azole-in-ring vs separate acidic group).
"""

import numpy as np
from rdkit import Chem

from features_pharmacophore_other_isoforms import _ACIDIC_PATTERNS
from features_pharmacophore_cyp1a2_v2 import _largest_fused_aromatic_system
from features_docked_electrostatics import _topology_only

_PYRIDINE_TYPE_N = Chem.MolFromSmarts("[n;H0;D2]")
_SULFONAMIDE_O = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[NX3]")
_OXIME = Chem.MolFromSmarts("[CX3]=[NX2][OX2]")

FEATURE_COLS = ["has_coord_group", "coord_atom_to_fe_dist", "ring_to_coord_dist"]


def _pose_mol_with_coords(pose_entry: dict):
    pose_mol = Chem.MolFromMol2Block(pose_entry["mol2_block"], sanitize=False, removeHs=False)
    if pose_mol is None:
        return None
    for atom in pose_mol.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
    try:
        pose_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(pose_mol)
        pose_mol = Chem.RemoveHs(pose_mol)
    except Exception:
        return None
    return pose_mol


def _coord_capable_atoms(mol: Chem.Mol) -> set[int]:
    idx = {m[0] for m in mol.GetSubstructMatches(_PYRIDINE_TYPE_N)}
    for pat in _ACIDIC_PATTERNS:
        for match in mol.GetSubstructMatches(pat):
            idx.update(match)
    for match in mol.GetSubstructMatches(_SULFONAMIDE_O):
        idx.update(match[1:3])  # the two =O atoms, not S or N
    for match in mol.GetSubstructMatches(_OXIME):
        idx.add(match[1])  # the sp2 N (the lone-pair-bearing, heme-ligating atom)
    return idx


def _empty_result():
    return (0.0, np.nan, np.nan)


def _one_compound(smiles: str, pose_entry: dict | None, fe: np.ndarray) -> tuple:
    if pose_entry is None:
        return _empty_result()

    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        return _empty_result()

    coord_idx = _coord_capable_atoms(orig_mol)
    has_coord = len(coord_idx) > 0
    if not has_coord:
        return (0.0, np.nan, np.nan)

    pose_mol = _pose_mol_with_coords(pose_entry)
    if pose_mol is None or pose_mol.GetNumAtoms() != orig_mol.GetNumAtoms():
        return (1.0, np.nan, np.nan)

    match = orig_mol.GetSubstructMatch(pose_mol)
    if len(match) != orig_mol.GetNumAtoms():
        match = orig_mol.GetSubstructMatch(pose_mol, useChirality=False)
    if len(match) != orig_mol.GetNumAtoms():
        try:
            match = _topology_only(orig_mol).GetSubstructMatch(_topology_only(pose_mol))
        except Exception:
            match = ()
    if len(match) != orig_mol.GetNumAtoms():
        return (1.0, np.nan, np.nan)
    orig_to_pose = {match[i]: i for i in range(len(match))}

    coords = pose_entry["coords"]
    n = min(len(pose_entry["elements"]), pose_mol.GetNumAtoms(), coords.shape[0])

    coord_pose_idx = [orig_to_pose[a] for a in coord_idx if a in orig_to_pose and orig_to_pose[a] < n]
    if not coord_pose_idx:
        return (1.0, np.nan, np.nan)
    dists_to_fe = np.linalg.norm(coords[coord_pose_idx] - fe, axis=1)
    best_local = int(np.argmin(dists_to_fe))
    coord_to_fe = float(dists_to_fe[best_local])
    coord_xyz = coords[coord_pose_idx[best_local]]

    ring_atoms = _largest_fused_aromatic_system(orig_mol)
    ring_to_coord = np.nan
    if ring_atoms is not None:
        ring_pose_idx = [orig_to_pose[a] for a in ring_atoms if a in orig_to_pose and orig_to_pose[a] < n]
        if ring_pose_idx:
            ring_centroid = coords[ring_pose_idx].mean(axis=0)
            ring_to_coord = float(np.linalg.norm(ring_centroid - coord_xyz))

    return (1.0, coord_to_fe, ring_to_coord)


def cyp2c9_pharmacophore_v2_matrix(molecule_names: list[str], smiles_list: list[str], poses: dict,
                                    fe_coords: dict) -> np.ndarray:
    rows = []
    n_ok = 0
    for name, smiles in zip(molecule_names, smiles_list):
        entry = poses.get(name)
        fe = fe_coords[entry["receptor"]] if entry is not None else None
        r = _one_compound(smiles, entry, fe) if entry is not None else _empty_result()
        if not np.isnan(r[1]):
            n_ok += 1
        rows.append(r)
    print(f"  [features_pharmacophore_cyp2c9_v2] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
