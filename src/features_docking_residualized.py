"""Heme-coordination docking features, residualized against MW/LogP.

Motivation (from the 2026-09-22 error-analysis session): raw docking energy
correlates with true CYP3A4 pIC50 (r=-0.35) largely because both energy and
potency independently correlate with molecular size/lipophilicity in this
dataset -- dividing by MW ("ligand efficiency") destroys almost all of that
correlation (r=-0.35 -> r=-0.03), confirming the raw signal is mostly a size
proxy, redundant with the MW/LogP the model already has via its RDKit
descriptors. A hard MW/LogP gate ("only use coordination for small molecules")
doesn't work either -- the compounds the model underpredicts (Group A) sit
right at the population's typical MW/LogP, they're only small *relative to
other potent compounds*, so a fixed absolute threshold flags ~half the dataset.

Instead: regress coord_fe_dist and coord_frac against MW + LogP across the
population (fit once, doesn't use the pIC50 label so this isn't leaky) and
keep the RESIDUAL -- "does this compound coordinate the heme iron better or
worse than its size/lipophilicity alone would predict" -- orthogonal to what
the model already has, rather than a redundant or overly-blunt transform.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RERANKED_RAW = PROJECT_ROOT / "models" / "gd2heme_reranked_raw.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv"

FEATURE_COLS = ["fe_dist_residual", "coord_frac_residual"]


def _build_per_compound_table() -> pd.DataFrame:
    raw = pd.read_csv(RERANKED_RAW)
    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5

    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    per_compound = best_per_receptor.groupby("Molecule_Name", as_index=False).agg(
        coord_fe_dist=("coord_fe_dist", "mean"),
        coord_frac=("coord_flag", "mean"),
    )
    return per_compound


def _residualize(per_compound: pd.DataFrame, smiles_lookup: dict) -> pd.DataFrame:
    per_compound = per_compound.copy()
    per_compound["MW"] = per_compound.Molecule_Name.map(
        lambda n: Descriptors.MolWt(Chem.MolFromSmiles(smiles_lookup[n]))
    )
    per_compound["LogP"] = per_compound.Molecule_Name.map(
        lambda n: Descriptors.MolLogP(Chem.MolFromSmiles(smiles_lookup[n]))
    )
    X = per_compound[["MW", "LogP"]].to_numpy()

    for target_col, resid_col in [("coord_fe_dist", "fe_dist_residual"), ("coord_frac", "coord_frac_residual")]:
        y = per_compound[target_col].to_numpy()
        reg = LinearRegression().fit(X, y)
        per_compound[resid_col] = y - reg.predict(X)

    return per_compound


def docking_residual_feature_matrix(molecule_names: list[str], smiles_list: list[str]) -> np.ndarray:
    """Returns (len(molecule_names), 2) array: [fe_dist_residual,
    coord_frac_residual]. Missing compounds mean-filled (residual mean ~0 by
    construction of OLS, so this is close to a neutral "no info" fill)."""
    smiles_lookup = dict(zip(molecule_names, smiles_list))
    per_compound = _build_per_compound_table()
    per_compound = _residualize(per_compound, smiles_lookup)
    lookup = per_compound.set_index("Molecule_Name")[FEATURE_COLS]
    means = lookup.mean()

    rows = []
    n_missing = 0
    for name in molecule_names:
        if name in lookup.index:
            rows.append(lookup.loc[name].to_numpy(dtype=float))
        else:
            rows.append(means.to_numpy(dtype=float))
            n_missing += 1
    if n_missing:
        print(f"  [features_docking_residualized] {n_missing}/{len(molecule_names)} compounds missing docking data, mean-filled")
    return np.array(rows, dtype=float)
