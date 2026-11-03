"""A real 3-point CYP1A2 pharmacophore (flat aromatic ring + hydrogen-bond
acceptor + hydrophobic group, per the literature model), distinct from the
earlier whole-molecule planarity descriptor (aromatic_fraction/n_aromatic_
rings/PBF) that tested null -- that measured overall molecular flatness, not
whether these three specific features exist and sit near each other. Measured
on the real docked pose (CYP1A2's own winning-pose extraction, coordination-
first selection), same lesson as every other isoform's pharmacophore this
session: bound geometry beats a free conformer.

Anchor is the largest FUSED aromatic ring system (not any single small ring)
-- CYP1A2 ligands are characteristically extended planar polyaromatics
(fluvoxamine's benzene+trifluoromethyl-phenoxy, ciprofloxacin's fused
quinolone, caffeine's fused purine), not just "has one aromatic ring".
"""

import numpy as np
from rdkit import Chem

from features_pharmacophore_kaur import _hydrophobic_fragments

_HBOND_ATOM = Chem.MolFromSmarts("[#7,#8]")
from features_docked_electrostatics import _topology_only  # noqa: E402

FEATURE_COLS = ["ring_size", "ring_to_hba_dist", "ring_to_hydrophobic_dist"]


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


def _largest_fused_aromatic_system(mol: Chem.Mol) -> list[int] | None:
    ri = mol.GetRingInfo()
    aromatic_rings = [
        set(ring) for ring in ri.AtomRings()
        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
    ]
    if not aromatic_rings:
        return None
    # merge rings sharing atoms (fused systems) via union-find over ring indices
    merged = [set(r) for r in aromatic_rings]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if merged[i] & merged[j]:
                    merged[i] |= merged[j]
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    return max(merged, key=len) if merged else None


def _empty_result():
    return (0.0, np.nan, np.nan)


def _one_compound(smiles: str, pose_entry: dict | None, fe: np.ndarray) -> tuple:
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
        return (ring_size, np.nan, np.nan)

    match = orig_mol.GetSubstructMatch(pose_mol)
    if len(match) != orig_mol.GetNumAtoms():
        match = orig_mol.GetSubstructMatch(pose_mol, useChirality=False)
    if len(match) != orig_mol.GetNumAtoms():
        try:
            match = _topology_only(orig_mol).GetSubstructMatch(_topology_only(pose_mol))
        except Exception:
            match = ()
    if len(match) != orig_mol.GetNumAtoms():
        return (ring_size, np.nan, np.nan)
    orig_to_pose = {match[i]: i for i in range(len(match))}

    coords = pose_entry["coords"]
    n = min(len(pose_entry["elements"]), pose_mol.GetNumAtoms(), coords.shape[0])

    ring_pose_idx = [orig_to_pose[a] for a in ring_atoms if a in orig_to_pose and orig_to_pose[a] < n]
    if not ring_pose_idx:
        return (ring_size, np.nan, np.nan)
    ring_centroid = coords[ring_pose_idx].mean(axis=0)

    # HBA atoms outside the aromatic ring system (exocyclic acceptors)
    hba_idx = {m[0] for m in orig_mol.GetSubstructMatches(_HBOND_ATOM)} - set(ring_atoms)
    hba_pose_idx = [orig_to_pose[a] for a in hba_idx if a in orig_to_pose and orig_to_pose[a] < n]
    ring_to_hba = np.nan
    if hba_pose_idx:
        dists = np.linalg.norm(coords[hba_pose_idx] - ring_centroid, axis=1)
        ring_to_hba = float(dists.min())

    # nearest non-aromatic hydrophobic fragment centroid
    ring_to_hydrophobic = np.nan
    for frag in _hydrophobic_fragments(orig_mol):
        frag_pose_idx = [orig_to_pose[a] for a in frag if a in orig_to_pose and orig_to_pose[a] < n]
        if not frag_pose_idx:
            continue
        centroid = coords[frag_pose_idx].mean(axis=0)
        d = float(np.linalg.norm(centroid - ring_centroid))
        if np.isnan(ring_to_hydrophobic) or d < ring_to_hydrophobic:
            ring_to_hydrophobic = d

    return (ring_size, ring_to_hba, ring_to_hydrophobic)


def cyp1a2_pharmacophore_v2_matrix(molecule_names: list[str], smiles_list: list[str], poses: dict,
                                    fe_coords: dict) -> np.ndarray:
    rows = []
    n_ok = 0
    for name, smiles in zip(molecule_names, smiles_list):
        entry = poses.get(name)
        fe = fe_coords[entry["receptor"]] if entry is not None else None
        r = _one_compound(smiles, entry, fe) if entry is not None else _empty_result()
        if not np.isnan(r[1]) or not np.isnan(r[2]):
            n_ok += 1
        rows.append(r)
    print(f"  [features_pharmacophore_cyp1a2_v2] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
