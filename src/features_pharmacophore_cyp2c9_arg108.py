"""CYP2C9 pharmacophore anchored on the literature's actual named residue --
Arg108's guanidinium group -- rather than the heme iron. Every CYP2C9
feature tried so far measured distance to Fe; the literature text says the
acidic group salt-bridges with "polar residues OR the ferric heme iron",
and Arg108 specifically is the classic named residue for this interaction
(warfarin, NSAIDs). This is a genuinely different anchor point, not a
variation on the Fe-distance theme.

Arg108 is present at the same residue number (108) in all 5 CYP2C9
receptors. Measures distance from the coordinating-capable group (same
definition as features_pharmacophore_cyp2c9_v2.py: azole/pyridine ring N,
acidic O, or sulfonamide O) to the guanidinium group's centroid (CZ/NH1/NH2),
on the real docked pose.
"""

import numpy as np
from rdkit import Chem

from features_pharmacophore_cyp2c9_v2 import _coord_capable_atoms, _pose_mol_with_coords
from features_docked_electrostatics import _topology_only

FEATURE_COLS = ["has_coord_group", "coord_atom_to_arg108_dist"]


def parse_arg108_guanidinium(receptor_pdb) -> np.ndarray:
    """Centroid of Arg108's guanidinium group (CZ, NH1, NH2)."""
    target_atoms = {"CZ", "NH1", "NH2"}
    coords = []
    with open(receptor_pdb) as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip()
        resnum = line[22:26].strip()
        atomname = line[12:16].strip()
        if resname == "ARG" and resnum == "108" and atomname in target_atoms:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"Arg108 guanidinium atoms not found in {receptor_pdb}")
    return np.array(coords).mean(axis=0)


def _empty_result():
    return (0.0, np.nan)


def _one_compound(smiles: str, pose_entry: dict | None, arg108_centroid: np.ndarray) -> tuple:
    if pose_entry is None:
        return _empty_result()

    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        return _empty_result()

    coord_idx = _coord_capable_atoms(orig_mol)
    has_coord = len(coord_idx) > 0
    if not has_coord:
        return (0.0, np.nan)

    pose_mol = _pose_mol_with_coords(pose_entry)
    if pose_mol is None or pose_mol.GetNumAtoms() != orig_mol.GetNumAtoms():
        return (1.0, np.nan)

    match = orig_mol.GetSubstructMatch(pose_mol)
    if len(match) != orig_mol.GetNumAtoms():
        match = orig_mol.GetSubstructMatch(pose_mol, useChirality=False)
    if len(match) != orig_mol.GetNumAtoms():
        try:
            match = _topology_only(orig_mol).GetSubstructMatch(_topology_only(pose_mol))
        except Exception:
            match = ()
    if len(match) != orig_mol.GetNumAtoms():
        return (1.0, np.nan)
    orig_to_pose = {match[i]: i for i in range(len(match))}

    coords = pose_entry["coords"]
    n = min(len(pose_entry["elements"]), pose_mol.GetNumAtoms(), coords.shape[0])

    coord_pose_idx = [orig_to_pose[a] for a in coord_idx if a in orig_to_pose and orig_to_pose[a] < n]
    if not coord_pose_idx:
        return (1.0, np.nan)
    dists = np.linalg.norm(coords[coord_pose_idx] - arg108_centroid, axis=1)
    return (1.0, float(dists.min()))


def cyp2c9_arg108_feature_matrix(molecule_names: list[str], smiles_list: list[str], poses: dict,
                                  arg108_coords: dict) -> np.ndarray:
    rows = []
    n_ok = 0
    for name, smiles in zip(molecule_names, smiles_list):
        entry = poses.get(name)
        arg108 = arg108_coords[entry["receptor"]] if entry is not None else None
        r = _one_compound(smiles, entry, arg108) if entry is not None else _empty_result()
        if not np.isnan(r[1]):
            n_ok += 1
        rows.append(r)
    print(f"  [features_pharmacophore_cyp2c9_arg108] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
