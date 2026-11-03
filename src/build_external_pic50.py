"""Merge ChEMBL + PubChem AID1851 into one leakage-checked external pIC50 table.

Drops any compound whose InChIKey matches TEST_BLINDED.csv (mandatory -- 1 compound
from each source was found to overlap the blind test set, confirming the earlier
concern that public bioactivity databases are enriched for FDA-approved/drug-like
compounds that overlap the challenge's FDAA test half). Compounds overlapping our OWN
training set are kept and flagged (not leakage -- those molecules are already labeled).

Where a compound has both a ChEMBL and a PubChem value for the same isoform, the two are
averaged (both are already curve-fitted/dose-response pIC50s, so a simple mean is a
reasonable consensus -- no evidence yet that one source is more reliable than the other
for this project).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("build_external_pic50")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def inchikey(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToInchiKey(mol) if mol else None


def main() -> None:
    test = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TEST_BLINDED.csv")
    train = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    test_keys = set(test["SMILES"].map(inchikey).dropna())
    train_keys = set(train["SMILES"].map(inchikey).dropna())

    chembl = pd.read_csv(EXTERNAL_DIR / "chembl_with_inchikey.csv")
    pubchem = pd.read_csv(EXTERNAL_DIR / "pubchem_with_inchikey.csv")

    chembl_leak = chembl["inchi_key"].isin(test_keys)
    pubchem_leak = pubchem["inchi_key"].isin(test_keys)
    log.info(f"Dropping {chembl_leak.sum()} ChEMBL rows and {pubchem_leak.sum()} PubChem rows "
             f"that overlap the blind test set")
    chembl = chembl[~chembl_leak].dropna(subset=["inchi_key"])
    pubchem = pubchem[~pubchem_leak].dropna(subset=["inchi_key"])

    # per-(inchi_key, isoform) median within each source first (some compounds have
    # multiple assay records), then combine sources by mean.
    chembl_agg = chembl.groupby(["inchi_key", "isoform"])["pic50"].median().rename("chembl_pic50")
    pubchem_agg = pubchem.groupby(["inchi_key", "isoform"])["pic50"].median().rename("pubchem_pic50")
    combined = pd.concat([chembl_agg, pubchem_agg], axis=1).reset_index()
    combined["pic50"] = combined[["chembl_pic50", "pubchem_pic50"]].mean(axis=1)
    combined["n_sources"] = combined[["chembl_pic50", "pubchem_pic50"]].notna().sum(axis=1)

    wide = combined.pivot(index="inchi_key", columns="isoform", values="pic50")
    wide = wide.reindex(columns=CYP_ISOFORMS)
    wide.columns = [f"{iso}_external_pic50" for iso in CYP_ISOFORMS]

    # keep one representative (canonical) SMILES per inchi_key, preferring ChEMBL's
    smiles_lookup = pd.concat([
        chembl[["inchi_key", "smiles"]], pubchem[["inchi_key", "smiles"]],
    ]).drop_duplicates(subset="inchi_key", keep="first").set_index("inchi_key")["smiles"]

    out = wide.join(smiles_lookup, how="left").reset_index()
    out["in_our_training_set"] = out["inchi_key"].isin(train_keys)
    out = out.dropna(subset=["smiles"])

    n_cols = [f"{iso}_external_pic50" for iso in CYP_ISOFORMS]
    has_any = out[n_cols].notna().any(axis=1)
    out = out[has_any]

    out_path = EXTERNAL_DIR / "external_pic50_clean.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Wrote {len(out):,} compounds -> {out_path}")
    log.info(f"  already in our training set (kept, flagged): {out['in_our_training_set'].sum():,}")
    for col in n_cols:
        log.info(f"  {col}: {out[col].notna().sum():,} compounds")


if __name__ == "__main__":
    main()
