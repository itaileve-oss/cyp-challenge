"""Like build_external_pic50.py, but keeps ChEMBL and PubChem/Veith(AID1851) as two
SEPARATE per-isoform columns instead of averaging them into one merged column. This
preserves the per-source structure that the merged version collapsed -- the two
sources are different labs/assay conditions, and averaging them into a single
"external_pic50" auxiliary target can reintroduce the exact cross-assay-bias problem
per-source auxiliary heads are meant to avoid (see cv_gnn_union_aux.py's 2026-09-15
result: macro ST-RAE 0.7485 using the merged column, worse than doing nothing).

Same leakage rule as build_external_pic50.py: drop any compound whose InChIKey matches
TEST_BLINDED.csv. Compounds overlapping our OWN training set are kept (not leakage).
"""

import logging
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("build_external_pic50_per_source")
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

    chembl_agg = chembl.groupby(["inchi_key", "isoform"])["pic50"].median().rename("pic50")
    pubchem_agg = pubchem.groupby(["inchi_key", "isoform"])["pic50"].median().rename("pic50")

    chembl_wide = chembl_agg.reset_index().pivot(index="inchi_key", columns="isoform", values="pic50")
    chembl_wide = chembl_wide.reindex(columns=CYP_ISOFORMS)
    chembl_wide.columns = [f"{iso}_chembl_pic50" for iso in CYP_ISOFORMS]

    pubchem_wide = pubchem_agg.reset_index().pivot(index="inchi_key", columns="isoform", values="pic50")
    pubchem_wide = pubchem_wide.reindex(columns=CYP_ISOFORMS)
    pubchem_wide.columns = [f"{iso}_pubchem_pic50" for iso in CYP_ISOFORMS]

    chembl_smiles = chembl.drop_duplicates(subset="inchi_key", keep="first").set_index("inchi_key")["smiles"]
    pubchem_smiles = pubchem.drop_duplicates(subset="inchi_key", keep="first").set_index("inchi_key")["smiles"]

    out = chembl_wide.join(pubchem_wide, how="outer")
    smiles_lookup = pd.concat([chembl_smiles, pubchem_smiles]).groupby(level=0).first()
    out = out.join(smiles_lookup.rename("smiles"), how="left").reset_index()
    out["in_our_training_set"] = out["inchi_key"].isin(train_keys)
    out = out.dropna(subset=["smiles"])

    n_cols = [f"{iso}_{src}_pic50" for iso in CYP_ISOFORMS for src in ("chembl", "pubchem")]
    has_any = out[n_cols].notna().any(axis=1)
    out = out[has_any]

    out_path = EXTERNAL_DIR / "external_pic50_per_source.csv"
    out.to_csv(out_path, index=False)
    log.info(f"Wrote {len(out):,} compounds -> {out_path}")
    log.info(f"  already in our training set (kept, flagged): {out['in_our_training_set'].sum():,}")
    for col in n_cols:
        log.info(f"  {col}: {out[col].notna().sum():,} compounds")


if __name__ == "__main__":
    main()
