"""Docked-pose version of the CYP2D6 basic-amine pharmacophore: measures the
literature's actual claim (basic-amine N sits 5-7A from the heme iron, the
oxidation site -- de Groot et al.) directly on the real docked pose, instead
of the free-conformer "distance to nearest aromatic ring" proxy in
features_pharmacophore_other_isoforms.py (written before CYP2D6 docking
existed). Same atom-mapping pattern as features_pharmacophore_kaur.py: match
the pose's own RDKit-parsed mol back to the training SMILES's mol to transfer
the reliable 2D substructure match (basic-amine SMARTS) onto the pose's 3D
coordinates.
"""

import numpy as np
from rdkit import Chem

from features_pharmacophore_other_isoforms import _BASIC_AMINE_N
from features_docked_electrostatics import _topology_only

FEATURE_COLS = ["has_basic_amine", "basic_n_to_fe_dist_docked"]


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


def _empty_result():
    return (0.0, np.nan)


def _one_compound(smiles: str, pose_entry: dict | None, fe: np.ndarray) -> tuple:
    if pose_entry is None:
        return _empty_result()

    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        return _empty_result()

    basic_n_idx = {m[0] for m in orig_mol.GetSubstructMatches(_BASIC_AMINE_N)}
    has_basic_n = len(basic_n_idx) > 0
    if not has_basic_n:
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
    # match[i] = orig-mol atom index for the i-th pose atom
    orig_to_pose = {match[i]: i for i in range(len(match))}

    coords = pose_entry["coords"]
    n = min(len(pose_entry["elements"]), pose_mol.GetNumAtoms(), coords.shape[0])

    pose_idx = [orig_to_pose[a] for a in basic_n_idx if a in orig_to_pose and orig_to_pose[a] < n]
    if not pose_idx:
        return (1.0, np.nan)

    dists = [float(np.linalg.norm(coords[i] - fe)) for i in pose_idx]
    return (1.0, min(dists))


def cyp2d6_docked_feature_matrix(molecule_names: list[str], smiles_list: list[str], poses: dict,
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
    print(f"  [features_pharmacophore_cyp2d6_docked] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
