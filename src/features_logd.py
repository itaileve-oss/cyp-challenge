"""A real pH-7.4 logD approximation and lipophilic-efficiency composite --
meant to be genuinely new information, not just a more-precise re-measurement
of the Crippen MolLogP already in the ~200-descriptor baseline block.

v2: uses dimorphite-dl's curated per-substructure pKa rules (12 functional
classes fit to experimental pKa data, dimorphite_dl/smarts/site_substructures.
smarts) to determine each compound's actual protonation state at pH 7.4,
precomputed once (precompute_dimorphite_protonation.py, dimorphite-dl only
lives in the GNINA conda env) and cached to models/dimorphite_ph74.csv. This
replaces the first pass's crude 2-class Henderson-Hasselbalch approximation
(one global pKa for "any basic amine", one for "any acid") -- Crippen logP
computed directly on the correctly-protonated structure, rather than a manual
post-hoc correction.

`logp_per_heavy_atom` is a ligand-efficiency-style composite (lipophilicity
normalized by size) that a shallow regularized tree (num_leaves=15) can't
easily reconstruct on its own from separate logP/heavy-atom-count columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "models" / "dimorphite_ph74.csv"

FEATURE_COLS = ["logd74", "logp_per_heavy_atom", "is_ionized_at_ph74"]

_cache = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        df = pd.read_csv(CACHE_PATH)
        _cache = dict(zip(df["SMILES"], df["protonated_smiles_ph74"]))
    return _cache


def _one_compound(smiles: str) -> tuple:
    cache = _load_cache()
    prot_smi = cache.get(smiles, smiles)
    mol = Chem.MolFromSmiles(prot_smi)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (0.0, 0.0, 0.0)

    logd = Crippen.MolLogP(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    logp_per_heavy = logd / n_heavy if n_heavy else 0.0
    is_ionized = 1.0 if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()) else 0.0
    return (float(logd), float(logp_per_heavy), is_ionized)


def logd_feature_matrix(smiles_list: list[str]) -> np.ndarray:
    rows = [_one_compound(smi) for smi in smiles_list]
    return np.array(rows, dtype=float)
