"""Featurization helpers for the CYP challenge baseline."""

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def morgan_fp_matrix(smiles_list: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """ECFP-equivalent Morgan fingerprints as a dense (n_mols, n_bits) uint8 array.

    A SMILES that fails to parse yields an all-zero row rather than raising, so
    callers can featurize a full column positionally without pre-filtering.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def rdkit_descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    """Full RDKit physchem/fragment descriptor set, columns with any NaN/inf dropped.

    Column count varies with the installed RDKit version (~200), which is fine since
    this is always consumed positionally alongside a fingerprint block, never by name.
    """
    calc = uru.RDKitDescriptors(hide_progress=True)
    desc_df = calc.pandas_smiles(smiles_list)
    clean_desc, _ = uru.clean_descriptors(desc_df)
    return np.asarray(clean_desc, dtype=float)
