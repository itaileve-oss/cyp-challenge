"""Docked-pose strain energy: MMFF94 energy of the ligand held in its docked
heavy-atom conformation (hydrogens added and locally relaxed, heavy atoms
frozen) minus the MMFF94 energy of its own globally-minimized free conformer.
A physics-based measure of "how much energetic penalty does this molecule pay
to adopt the shape the pocket forces on it" -- high strain should correlate
with weaker binding, independent of anything already in fp+desc or the
existing docking/coordination features (those measure fit and heme
engagement, not conformational cost).

Not tried anywhere in this project before now.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from features_docked_electrostatics import _topology_only

FEATURE_COLS = ["strain_energy", "docked_mmff_energy", "global_min_mmff_energy"]

SEED = 42


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


def _docked_conformation_energy(orig_mol: Chem.Mol, docked_coords: np.ndarray) -> float | None:
    """Build orig_mol (reliable bond orders/aromaticity) with a conformer set to the
    docked heavy-atom positions, add Hs with sensible initial coordinates, relax ONLY
    the hydrogens (heavy atoms frozen via a strong position constraint), then return
    the resulting MMFF94 energy -- the energy of the docked heavy-atom framework with
    locally-appropriate hydrogens, not a globally re-optimized structure.
    """
    mol = Chem.Mol(orig_mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, docked_coords[i].tolist())
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    mol_h = Chem.AddHs(mol, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(mol_h)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props)
    if ff is None:
        return None
    # freeze heavy atoms, let only the newly-added H positions relax
    n_heavy = mol.GetNumAtoms()
    for i in range(n_heavy):
        ff.MMFFAddPositionConstraint(i, 0.0, 1.0e5)
    ff.Minimize(maxIts=500)
    return float(ff.CalcEnergy())


def _global_min_energy(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.useRandomCoords = True
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return None
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    if ff is None:
        return None
    ff.Minimize(maxIts=1000)
    return float(ff.CalcEnergy())


def _empty_result():
    return (np.nan, np.nan, np.nan)


def _one_compound(smiles: str, pose_entry: dict | None) -> tuple:
    if pose_entry is None:
        return _empty_result()

    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        return _empty_result()

    pose_mol = _pose_mol_with_coords(pose_entry)
    if pose_mol is None or pose_mol.GetNumAtoms() != orig_mol.GetNumAtoms():
        return _empty_result()

    match = orig_mol.GetSubstructMatch(pose_mol)
    if len(match) != orig_mol.GetNumAtoms():
        match = orig_mol.GetSubstructMatch(pose_mol, useChirality=False)
    if len(match) != orig_mol.GetNumAtoms():
        try:
            match = _topology_only(orig_mol).GetSubstructMatch(_topology_only(pose_mol))
        except Exception:
            match = ()
    if len(match) != orig_mol.GetNumAtoms():
        return _empty_result()
    # match[i] = orig-mol atom index for pose atom i; build orig-indexed coords
    pose_coords = pose_entry["coords"]
    n = min(len(match), pose_mol.GetNumAtoms(), pose_coords.shape[0])
    orig_coords = np.zeros((orig_mol.GetNumAtoms(), 3))
    filled = np.zeros(orig_mol.GetNumAtoms(), dtype=bool)
    for i in range(n):
        orig_coords[match[i]] = pose_coords[i]
        filled[match[i]] = True
    if not filled.all():
        return _empty_result()

    try:
        docked_e = _docked_conformation_energy(orig_mol, orig_coords)
        global_e = _global_min_energy(smiles)
    except Exception:
        return _empty_result()
    if docked_e is None or global_e is None:
        return _empty_result()

    strain = docked_e - global_e
    return (strain, docked_e, global_e)


def strain_energy_feature_matrix(molecule_names: list[str], smiles_list: list[str], poses: dict) -> np.ndarray:
    rows = []
    n_ok = 0
    for i, (name, smiles) in enumerate(zip(molecule_names, smiles_list), 1):
        entry = poses.get(name)
        r = _one_compound(smiles, entry) if entry is not None else _empty_result()
        if not np.isnan(r[0]):
            n_ok += 1
        rows.append(r)
        if i % 200 == 0:
            print(f"  [features_strain_energy] {i}/{len(molecule_names)} (ok={n_ok})", flush=True)
    print(f"  [features_strain_energy] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
