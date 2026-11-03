"""Extends the CYP1A2 3-point pharmacophore (features_pharmacophore_cyp1a2_v2.py:
flat aromatic ring + HBA + hydrophobic group) with a 4th point: nearest
hydrogen-bond DONOR distance to the ring. The literature description only
named an acceptor, but donor-mediated backbone H-bonds are common in real
CYP450 pockets and cost nothing extra to test now that the ring-anchor +
atom-mapping machinery already exists.
"""

import numpy as np
from rdkit import Chem

from features_pharmacophore_kaur import _hydrophobic_fragments
from features_pharmacophore_cyp1a2_v2 import _largest_fused_aromatic_system, _pose_mol_with_coords
from features_docked_electrostatics import _topology_only

_HBOND_ACCEPTOR = Chem.MolFromSmarts("[#7,#8]")
_HBOND_DONOR = Chem.MolFromSmarts("[OX2H,NX3H1,NX3H2;!$(NC=O)]")

FEATURE_COLS = ["ring_size", "ring_to_hba_dist", "ring_to_hydrophobic_dist", "ring_to_hbd_dist"]


def _empty_result():
    return (0.0, np.nan, np.nan, np.nan)


def _one_compound(smiles: str, pose_entry: dict | None) -> tuple:
    if pose_entry is None:
        return _empty_result()

    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        return _empty_result()

    ring_atoms = _largest_fused_aromatic_system(orig_mol)
    if ring_atoms is None:
        return _empty_result()
    ring_size = float(len(ring_atoms))

    pose_mol = _pose_mol_with_coords(pose_entry)
    if pose_mol is None or pose_mol.GetNumAtoms() != orig_mol.GetNumAtoms():
        return (ring_size, np.nan, np.nan, np.nan)

    match = orig_mol.GetSubstructMatch(pose_mol)
    if len(match) != orig_mol.GetNumAtoms():
        match = orig_mol.GetSubstructMatch(pose_mol, useChirality=False)
    if len(match) != orig_mol.GetNumAtoms():
        try:
            match = _topology_only(orig_mol).GetSubstructMatch(_topology_only(pose_mol))
        except Exception:
            match = ()
    if len(match) != orig_mol.GetNumAtoms():
        return (ring_size, np.nan, np.nan, np.nan)
    orig_to_pose = {match[i]: i for i in range(len(match))}

    coords = pose_entry["coords"]
    n = min(len(pose_entry["elements"]), pose_mol.GetNumAtoms(), coords.shape[0])

    ring_pose_idx = [orig_to_pose[a] for a in ring_atoms if a in orig_to_pose and orig_to_pose[a] < n]
    if not ring_pose_idx:
        return (ring_size, np.nan, np.nan, np.nan)
    ring_centroid = coords[ring_pose_idx].mean(axis=0)

    hba_idx = {m[0] for m in orig_mol.GetSubstructMatches(_HBOND_ACCEPTOR)} - set(ring_atoms)
    hba_pose_idx = [orig_to_pose[a] for a in hba_idx if a in orig_to_pose and orig_to_pose[a] < n]
    ring_to_hba = np.nan
    if hba_pose_idx:
        ring_to_hba = float(np.linalg.norm(coords[hba_pose_idx] - ring_centroid, axis=1).min())

    ring_to_hydrophobic = np.nan
    for frag in _hydrophobic_fragments(orig_mol):
        frag_pose_idx = [orig_to_pose[a] for a in frag if a in orig_to_pose and orig_to_pose[a] < n]
        if not frag_pose_idx:
            continue
        centroid = coords[frag_pose_idx].mean(axis=0)
        d = float(np.linalg.norm(centroid - ring_centroid))
        if np.isnan(ring_to_hydrophobic) or d < ring_to_hydrophobic:
            ring_to_hydrophobic = d

    hbd_idx = {m[0] for m in orig_mol.GetSubstructMatches(_HBOND_DONOR)} - set(ring_atoms)
    hbd_pose_idx = [orig_to_pose[a] for a in hbd_idx if a in orig_to_pose and orig_to_pose[a] < n]
    ring_to_hbd = np.nan
    if hbd_pose_idx:
        ring_to_hbd = float(np.linalg.norm(coords[hbd_pose_idx] - ring_centroid, axis=1).min())

    return (ring_size, ring_to_hba, ring_to_hydrophobic, ring_to_hbd)


def cyp1a2_pharmacophore_v3_matrix(molecule_names: list[str], smiles_list: list[str], poses: dict) -> np.ndarray:
    rows = []
    n_ok = 0
    for name, smiles in zip(molecule_names, smiles_list):
        entry = poses.get(name)
        r = _one_compound(smiles, entry) if entry is not None else _empty_result()
        if any(not np.isnan(v) for v in r[1:]):
            n_ok += 1
        rows.append(r)
    print(f"  [features_pharmacophore_cyp1a2_v3] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
