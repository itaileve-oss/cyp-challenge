"""Per-atom electrostatic features from GFN2-xTB, take 2 -- rotation-invariant only.

The earlier attempt (features_multipole.py: charge + raw dipole xyz + raw
quadrupole tensor, see project memory) made the GNN worse. Likely cause: the raw
dipole (x,y,z) and quadrupole tensor components are NOT rotation-invariant -- they
depend on the arbitrary orientation of the single ETKDG conformer embedded per
molecule, with no canonical frame. Charge is a scalar and was fine on its own
(only a small regression); the vector/tensor components are the suspect part.

This version keeps only scalar, rotation-invariant quantities:
  - partial charge (already scalar/invariant)
  - dipole MAGNITUDE (||dipole vector||, invariant -- not the raw x/y/z)
  - static atomic polarizability alpha(0), from xtb's D4-dispersion population
    analysis table (same physical model that gives C6/C8 dispersion coefficients;
    alpha(0) here is the isotropic/spherically-averaged polarizability, so it's
    inherently orientation-independent, no fix needed)
Quadrupole is dropped entirely -- a proper rotation-invariant reduction of it
(eigenvalues, trace-based anisotropy) is more involved and not attempted here.

Same xtb_env conda env, same `xtb ... --gfn 2 --pop --json` call as
features_multipole.py -- charge/dipole still come from xtbout.json, alpha(0) is
parsed from stdout text (not exposed in the JSON), no extra xtb invocation needed.
"""

import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

XTB_BIN = str(Path.home() / "miniconda3" / "envs" / "xtb_env" / "bin" / "xtb")
N_ELECTROSTATIC_FEATURES = 3  # charge, dipole magnitude, atomic polarizability

_ALPHA_ROW_RE = re.compile(
    r"^\s*\d+\s+\d+\s+[A-Za-z]+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)\s*$"
)


def _embed_xyz(smiles: str, xyz_path: Path, seed: int = 42) -> tuple[bool, int]:
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


def _parse_atomic_polarizabilities(stdout_text: str, n_atoms_total: int) -> list[float]:
    """Parse the "#  Z  sym  covCN  q  C6AA  α(0)" population-analysis table.
    Returns alpha(0) per atom, in input-file atom order. Raises if the table isn't
    found or doesn't have the expected row count -- caller catches and falls back.
    """
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


def _run_one(args: tuple[int, str]) -> tuple[int, np.ndarray]:
    idx, smiles = args
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        xyz_path = tmpdir_p / "mol.xyz"
        ok, n_heavy = _embed_xyz(smiles, xyz_path)
        n_heavy = max(n_heavy, 1)
        fallback = np.zeros((n_heavy, N_ELECTROSTATIC_FEATURES), dtype=np.float32)
        if not ok:
            return idx, fallback

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        try:
            result = subprocess.run(
                [XTB_BIN, "mol.xyz", "--gfn", "2", "--pop", "--json"],
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
            feat = np.stack([charges, dipole_mag, alphas], axis=1)[:n_heavy]
            if feat.shape != (n_heavy, N_ELECTROSTATIC_FEATURES):
                return idx, fallback
        except Exception:
            return idx, fallback
        return idx, feat


def electrostatic_features(smiles_list: list[str], n_workers: int = 8) -> list[np.ndarray]:
    """Per-molecule, per-heavy-atom (charge, dipole magnitude, polarizability)
    arrays, all rotation-invariant scalars. Ragged (one (n_atoms_i, 3) array per
    molecule). Molecules where embedding/xtb/parsing fails get an all-zero array.
    """
    results: list[np.ndarray] = [None] * len(smiles_list)  # type: ignore[list-item]
    tasks = list(enumerate(smiles_list))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for idx, feat in pool.map(_run_one, tasks):
            results[idx] = feat
    return results
