"""Semi-empirical quantum descriptors (GFN2-xTB) -- physics-based, not empirical/graph.

For each molecule: embed a single MMFF-optimized 3D conformer (RDKit ETKDG), run a
GFN2-xTB single-point calculation, and parse out frontier-orbital and electrostatic
properties. These describe electronic structure (where electrons actually sit, how
easily the molecule gives/accepts them) rather than shape or substructure counts --
relevant to CYP450 chemistry, which is fundamentally an electron-transfer/oxidation
process at the heme iron.

Requires the `xtb_env` conda environment (GFN2-xTB binary, no GPU needed).
"""

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
FEATURE_NAMES = [
    "xtb_homo_ev",
    "xtb_lumo_ev",
    "xtb_gap_ev",
    "xtb_electronegativity",
    "xtb_hardness",
    "xtb_dipole_debye",
    "xtb_total_energy_ha",
    "xtb_dispersion_ha",
    "xtb_isotropic_es_ha",
]
N_XTB_FEATURES = len(FEATURE_NAMES)

_HOMO_RE = re.compile(r"^\s*\d+\s+[\d.]+\s+[-\d.]+\s+([-\d.]+)\s+\(HOMO\)", re.MULTILINE)
_LUMO_RE = re.compile(r"^\s*\d+\s+[-\d.]+\s+([-\d.]+)\s+\(LUMO\)", re.MULTILINE)
_GAP_RE = re.compile(r"::\s*HOMO-LUMO gap\s+([-\d.]+) eV")
_ENERGY_RE = re.compile(r"::\s*total energy\s+([-\d.]+) Eh")
_DISPERSION_RE = re.compile(r"::\s*->\s*dispersion\s+([-\d.]+) Eh")
_ISOTROPIC_ES_RE = re.compile(r"::\s*->\s*isotropic ES\s+([-\d.]+) Eh")
_DIPOLE_RE = re.compile(r"^\s*full:\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+([\d.]+)", re.MULTILINE)


def _embed_xyz(smiles: str, xyz_path: Path, seed: int = 42) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    try:
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return False
        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=500)
        Chem.MolToXYZFile(mol, str(xyz_path), confId=cid)
    except Exception:
        return False
    return True


def _parse_xtb_output(text: str) -> np.ndarray:
    out = np.full(N_XTB_FEATURES, np.nan)
    homo_m = _HOMO_RE.search(text)
    lumo_m = _LUMO_RE.search(text)
    gap_m = _GAP_RE.search(text)
    energy_m = _ENERGY_RE.search(text)
    disp_m = _DISPERSION_RE.search(text)
    iso_m = _ISOTROPIC_ES_RE.search(text)
    dip_m = _DIPOLE_RE.search(text)

    homo = float(homo_m.group(1)) if homo_m else np.nan
    lumo = float(lumo_m.group(1)) if lumo_m else np.nan
    out[0] = homo
    out[1] = lumo
    out[2] = float(gap_m.group(1)) if gap_m else np.nan
    out[3] = -(homo + lumo) / 2 if np.isfinite(homo) and np.isfinite(lumo) else np.nan
    out[4] = (lumo - homo) / 2 if np.isfinite(homo) and np.isfinite(lumo) else np.nan
    out[5] = float(dip_m.group(1)) if dip_m else np.nan
    out[6] = float(energy_m.group(1)) if energy_m else np.nan
    out[7] = float(disp_m.group(1)) if disp_m else np.nan
    out[8] = float(iso_m.group(1)) if iso_m else np.nan
    return out


def _run_one(args: tuple[int, str]) -> tuple[int, np.ndarray]:
    idx, smiles = args
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        xyz_path = tmpdir_p / "mol.xyz"
        if not _embed_xyz(smiles, xyz_path):
            return idx, np.full(N_XTB_FEATURES, np.nan)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        try:
            result = subprocess.run(
                [XTB_BIN, "mol.xyz", "--gfn", "2"],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return idx, np.full(N_XTB_FEATURES, np.nan)
        return idx, _parse_xtb_output(result.stdout)


def xtb_descriptor_matrix(smiles_list: list[str], n_workers: int = 8) -> np.ndarray:
    """GFN2-xTB descriptors for a list of SMILES, computed in parallel processes."""
    out = np.full((len(smiles_list), N_XTB_FEATURES), np.nan)
    tasks = list(enumerate(smiles_list))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for idx, vec in pool.map(_run_one, tasks):
            out[idx] = vec
    return out
