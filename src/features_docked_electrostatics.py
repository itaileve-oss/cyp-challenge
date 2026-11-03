"""Per-atom electrostatic features (charge, dipole magnitude, polarizability --
same rotation-invariant trio as features_electrostatics.py) computed by GFN2-xTB
on the actual DOCKED pose geometry, not a free-ligand MMFF-minimized conformer.

Every earlier electrostatic-feature attempt on this project used a conformer that
had never seen the receptor -- the free-ligand lowest-energy shape, which is often
NOT the bound conformation. This is the first version computed on coordinates
extracted directly from a real docked pose (the coord_first-selected,
heme-coordination-prioritized winning pose per compound, see
extract_winning_poses.py), sidestepping that specific, previously-diagnosed
mismatch. CYP3A4-only -- docking only exists for CYP3A4-labeled compounds.

Two remapping steps this needs that the free-conformer version didn't:
  1. Atom order: the docked pose comes from a dimorphite-protonated SMILES
     variant prepared independently (prepare_ligands_gd2heme_v2.py) and written
     through OpenBabel -- its heavy-atom order does NOT reliably match
     Chem.MolFromSmiles(training_smiles)'s order (checked empirically: ~40%
     mismatch in a 5-compound sample). Remapped via GetSubstructMatch between the
     pose's own mol2-derived Mol and the training-SMILES Mol.
  2. Net formal charge: the docked species is protonated at pH 7.4 and may carry
     a nonzero net charge (e.g. a protonated amine). xtb needs --chrg explicitly
     for this -- earlier scripts on this project embedded from a normally-neutral
     training SMILES and never passed --chrg, which would have been silently
     wrong here given genuinely protonated structures.
"""

import json
import os
import pickle
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem

XTB_BIN = str(Path.home() / "miniconda3" / "envs" / "xtb_env" / "bin" / "xtb")
N_ELECTROSTATIC_FEATURES = 3  # charge, dipole magnitude, atomic polarizability
WINNING_POSES_PKL = Path("/home/ileven/Projects/cyp-challenge/models/winning_docked_poses.pkl")

_ALPHA_ROW_RE = re.compile(
    r"^\s*\d+\s+\d+\s+[A-Za-z]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)\s*$"
)


def _topology_only(mol: Chem.Mol) -> Chem.Mol:
    """Strip bond order/aromaticity down to plain connectivity (all bonds ->
    SINGLE, no aromatic flags). OpenBabel's SYBYL bond typing (used when writing
    the docked-pose mol2) and RDKit's own kekulization from SMILES frequently
    disagree on bond order/aromaticity for the exact same molecule, which makes
    a normal (bond-order-sensitive) GetSubstructMatch fail even when the atoms
    and connectivity are identical -- matching on topology alone sidesteps that."""
    # Formal charge is deliberately left untouched: zeroing it while keeping the
    # same bond count breaks valence for atoms whose charge exists specifically
    # to support that bonding pattern (e.g. a protonated ring N with 4 bonds).
    # Skip full sanitization entirely -- plain substructure isomorphism only
    # needs elements/bonds/degree set, not a chemically valid, valence-checked
    # molecule, and valence checking is exactly what a differently-protonated
    # pose vs. the neutral training SMILES would otherwise trip over.
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumRadicalElectrons(0)
        atom.SetNumExplicitHs(0)
        atom.SetFormalCharge(0)  # protonation state differences shouldn't block a topology match
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)  # strict=False: tolerates the now-inconsistent valence/charge combo
    Chem.FastFindRings(m)
    return m


def _prepare_docked_xyz(pose_entry: dict, orig_smiles: str, xyz_path: Path):
    """Returns (ok, n_heavy_orig, match, charge) where match[i] = orig_mol atom
    index for the i-th heavy atom in the docked pose's own atom order (None on
    any failure)."""
    pose_mol = Chem.MolFromMol2Block(pose_entry["mol2_block"], sanitize=False, removeHs=False)
    if pose_mol is None:
        return False, 0, None, 0
    # Heavy-atom-only mol2 with no explicit Hs: RDKit's sanitizer otherwise fills
    # unmet valence with radical electrons instead of implicit H (same class of
    # bug documented in prepare_query_structures.py -- fix is identical). Also,
    # removeHs on parse silently no-ops when sanitize=False, so strip explicit Hs
    # ourselves after sanitizing (there normally aren't any in this heavy-atom-only
    # mol2, but a couple of stray ones have shown up -- drop them explicitly
    # rather than let them desync the atom count from orig_mol).
    for atom in pose_mol.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
    try:
        pose_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(pose_mol)
        pose_mol = Chem.RemoveHs(pose_mol)
    except Exception:
        return False, 0, None, 0
    orig_mol = Chem.MolFromSmiles(orig_smiles)
    if orig_mol is None:
        return False, 0, None, 0
    n_heavy_orig = orig_mol.GetNumAtoms()
    if pose_mol.GetNumAtoms() != n_heavy_orig:
        return False, n_heavy_orig, None, 0

    match = orig_mol.GetSubstructMatch(pose_mol)
    if len(match) != n_heavy_orig:
        match = orig_mol.GetSubstructMatch(pose_mol, useChirality=False)
    if len(match) != n_heavy_orig:
        try:
            match = _topology_only(orig_mol).GetSubstructMatch(_topology_only(pose_mol))
        except Exception:
            match = ()
    if len(match) != n_heavy_orig:
        return False, n_heavy_orig, None, 0

    try:
        pose_mol_h = Chem.AddHs(pose_mol, addCoords=True)
        charge = Chem.GetFormalCharge(pose_mol_h)
        Chem.MolToXYZFile(pose_mol_h, str(xyz_path))
    except Exception:
        return False, n_heavy_orig, None, 0
    return True, n_heavy_orig, match, charge


def _parse_atomic_polarizabilities(stdout_text: str, n_atoms_total: int) -> list[float]:
    lines = stdout_text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if "α(0)" in line)
    alphas = []
    for line in lines[header_idx + 1 : header_idx + 1 + n_atoms_total]:
        m = _ALPHA_ROW_RE.match(line)
        if not m:
            raise ValueError(f"Unexpected population table row: {line!r}")
        alphas.append(float(m.group(1)))
    if len(alphas) != n_atoms_total:
        raise ValueError(f"Expected {n_atoms_total} polarizability rows, got {len(alphas)}")
    return alphas


def _run_one(args: tuple[int, str, dict | None]) -> tuple[int, np.ndarray]:
    idx, smiles, pose_entry = args
    orig_mol = Chem.MolFromSmiles(smiles) if smiles else None
    n_heavy_orig = orig_mol.GetNumAtoms() if orig_mol is not None else 1
    fallback = np.zeros((max(n_heavy_orig, 1), N_ELECTROSTATIC_FEATURES), dtype=np.float32)
    if pose_entry is None:
        return idx, fallback

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        xyz_path = tmpdir_p / "mol.xyz"
        ok, n_heavy_orig, match, charge = _prepare_docked_xyz(pose_entry, smiles, xyz_path)
        if not ok:
            return idx, fallback

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        try:
            result = subprocess.run(
                [XTB_BIN, "mol.xyz", "--gfn", "2", "--chrg", str(charge), "--pop", "--json"],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            with open(tmpdir_p / "xtbout.json") as f:
                data = json.load(f)
            charges = np.array(data["partial charges"], dtype=np.float32)
            dipoles = np.array(data["atomic dipole moments"], dtype=np.float32)
            dipole_mag = np.linalg.norm(dipoles, axis=1)
            n_total = len(charges)
            alphas = np.array(
                _parse_atomic_polarizabilities(result.stdout, n_total), dtype=np.float32
            )
            n_heavy_pose = len(match)
            feat_pose_order = np.stack([charges, dipole_mag, alphas], axis=1)[:n_heavy_pose]
            if feat_pose_order.shape != (n_heavy_pose, N_ELECTROSTATIC_FEATURES):
                return idx, fallback

            feat_orig_order = np.zeros((n_heavy_orig, N_ELECTROSTATIC_FEATURES), dtype=np.float32)
            for pose_i, orig_i in enumerate(match):
                feat_orig_order[orig_i] = feat_pose_order[pose_i]
        except Exception:
            return idx, fallback
        return idx, feat_orig_order


def docked_electrostatic_features(
    molecule_names: list[str], smiles_list: list[str], n_workers: int = 8
) -> list[np.ndarray]:
    """Per-molecule, per-heavy-atom (charge, dipole magnitude, polarizability)
    arrays computed on the docked pose, remapped to Chem.MolFromSmiles(smiles)
    atom order. Compounds with no docked pose (not CYP3A4, or the 1/2335 that
    failed docking) get an all-zero array."""
    with open(WINNING_POSES_PKL, "rb") as f:
        poses = pickle.load(f)

    tasks = [
        (i, smi, poses.get(name))
        for i, (name, smi) in enumerate(zip(molecule_names, smiles_list))
    ]
    n_with_pose = sum(1 for _, _, p in tasks if p is not None)
    print(f"  [features_docked_electrostatics] {n_with_pose}/{len(tasks)} compounds have a docked pose", flush=True)

    results: list[np.ndarray] = [None] * len(tasks)  # type: ignore[list-item]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for idx, feat in pool.map(_run_one, tasks):
            results[idx] = feat
    n_ok = sum(1 for r in results if r is not None and r.any())
    print(f"  [features_docked_electrostatics] xtb succeeded for {n_ok}/{len(tasks)}", flush=True)
    return results
