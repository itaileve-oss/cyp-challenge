"""Loads external_pic50_clean.csv (ChEMBL + PubChem AID1851, leakage-checked against
TEST_BLINDED.csv -- see build_external_pic50.py) into a per-molecule, per-isoform
auxiliary regression target for multi-task co-training, keyed by InChIKey since these
compounds live outside our own Molecule_Name namespace.

Same rationale as features_single_conc.py's log2fc head, but a much bigger expansion:
~25,600 net-new molecules (after dedup against our own training set and the blind test)
carry a real dose-response-fitted pIC50 for at least one isoform, vs. our own 1,285-2,335
labeled compounds per isoform. These can't be pooled directly as primary-head training
rows (different labs/assay conditions than OpenADMET's own data -- a real, if smaller,
version of the same assay-heterogeneity concern that ruled out naively pooling TDC/Veith
data earlier), but work as a second auxiliary co-training head exactly like log2fc does.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToInchiKey(mol) if mol else None


def build_external_pic50_matrix(smiles_list: list[str], external_path: Path) -> np.ndarray:
    """Returns (N, 4) z-scored (per-isoform) external_pic50 matrix, NaN where a molecule
    (matched by InChIKey) has no external measurement for that isoform.
    """
    ext = pd.read_csv(external_path)
    ext = ext.set_index("inchi_key")

    keys = [inchikey(s) for s in smiles_list]
    matrix = np.full((len(smiles_list), len(CYP_ISOFORMS)), np.nan)

    for iso_idx, iso in enumerate(CYP_ISOFORMS):
        col = f"{iso}_external_pic50"
        if col not in ext.columns:
            continue
        vals = ext[col].dropna()
        mean, std = vals.mean(), vals.std()
        z = (vals - mean) / std
        for row, key in enumerate(keys):
            if key is not None and key in z.index:
                matrix[row, iso_idx] = z.loc[key] if np.isscalar(z.loc[key]) else z.loc[key].iloc[0]

    return matrix


def load_external_only_compounds(external_path: Path, exclude_keys: set[str]) -> pd.DataFrame:
    """External compounds NOT already in our training set -- the net-new molecules that
    expand the union pool. `exclude_keys` should be our own training set's InChIKeys.
    """
    ext = pd.read_csv(external_path)
    return ext[~ext["inchi_key"].isin(exclude_keys)].reset_index(drop=True)
