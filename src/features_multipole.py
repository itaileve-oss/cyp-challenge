"""Per-atom multipole moments (charge, dipole, quadrupole) from GFN2-xTB's native
CAMM (Cumulative Atomic Multipole Moment) model.

GFN2-xTB's anisotropic electrostatics (AES) term carries atom-centered dipole and
quadrupole moments as part of the Hamiltonian itself -- this is the actual physical
difference between GFN2 and GFN1 (which only had point charges). So this isn't a
separate DFT-density-partitioning scheme (MBIS/DDEC/CHELPG) bolted on after the
fact; it's read straight out of the same single-point calculation via `xtb --json`,
at the same per-molecule cost already used for the whole-molecule descriptors in
features_xtb.py.

Requires the `xtb_env` conda environment (GFN2-xTB binary).

Atom ordering: RDKit's Chem.AddHs() appends new explicit hydrogens *after* all
existing heavy-atom indices, so the first n_heavy entries of xtb's per-atom output
arrays line up exactly, in order, with the heavy-atom-only graph built by
features_graph.py (which never adds explicit Hs). We slice off everything past
n_heavy and never touch hydrogen entries.
"""

import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

XTB_BIN = str(Path.home() / "miniconda3" / "envs" / "xtb_env" / "bin" / "xtb")
N_MULTIPOLE_FEATURES = 1 + 3 + 6  # charge + dipole(x,y,z) + quadrupole(6 unique components)


def _embed_xyz(smiles: str, xyz_path: Path, seed: int = 42) -> tuple[bool, int]:
    """Embed a 3D conformer (with explicit Hs, required by xtb) and write it out.
    Returns (success, n_heavy_atoms) -- n_heavy_atoms is needed by the caller even
    on failure, to size the zero-filled fallback array correctly.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, 0
    n_heavy = mol.GetNumAtoms()
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    try:
        cid = AllChem.EmbedMolecule(mol_h, params)
        if cid < 0:
            return False, n_heavy
        AllChem.MMFFOptimizeMolecule(mol_h, confId=cid, maxIters=500)
        Chem.MolToXYZFile(mol_h, str(xyz_path), confId=cid)
    except Exception:
        return False, n_heavy
    return True, n_heavy


def _run_one(args: tuple[int, str]) -> tuple[int, np.ndarray]:
    idx, smiles = args
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        xyz_path = tmpdir_p / "mol.xyz"
        ok, n_heavy = _embed_xyz(smiles, xyz_path)
        n_heavy = max(n_heavy, 1)
        fallback = np.zeros((n_heavy, N_MULTIPOLE_FEATURES), dtype=np.float32)
        if not ok:
            return idx, fallback

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        try:
            subprocess.run(
                [XTB_BIN, "mol.xyz", "--gfn", "2", "--pop", "--json"],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            with open(tmpdir_p / "xtbout.json") as f:
                data = json.load(f)
            charges = np.array(data["partial charges"][:n_heavy], dtype=np.float32).reshape(-1, 1)
            dipoles = np.array(data["atomic dipole moments"][:n_heavy], dtype=np.float32)
            quads = np.array(data["atomic quadrupole moments"][:n_heavy], dtype=np.float32)
            feat = np.hstack([charges, dipoles, quads])
            if feat.shape != (n_heavy, N_MULTIPOLE_FEATURES):
                return idx, fallback
        except Exception:
            return idx, fallback
        return idx, feat


def multipole_features(smiles_list: list[str], n_workers: int = 8) -> list[np.ndarray]:
    """Per-molecule, per-heavy-atom multipole feature arrays (charge + dipole +
    quadrupole, 10 columns), computed in parallel. Ragged -- one (n_atoms_i, 10)
    array per molecule, since atom counts vary. Molecules where embedding or xtb
    fails get an all-zero array (not NaN/dropped) so downstream code never has to
    special-case them.
    """
    results: list[np.ndarray] = [None] * len(smiles_list)  # type: ignore[list-item]
    tasks = list(enumerate(smiles_list))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for idx, feat in pool.map(_run_one, tasks):
            results[idx] = feat
    return results
