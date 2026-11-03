"""Pull the Veith CYP inhibition panel (PubChem AID 1851) as auxiliary pretraining data.

Adapted from SuperCowPowers/workbench's pull_pubchem_cyp_data.py (public REST API, no
credentials needed -- fully portable, unlike their ChEMBL pull which reads a private S3
mirror). A qHTS 15-point dose-response screen of ~17k compounds against 5 human CYP
isoforms (we keep the 4 our challenge scores: 1A2, 2C9, 2D6, 3A4). Fit_LogAC50 is
log10(AC50) in molar, so pIC50 = -Fit_LogAC50 -- a real dose-response-fitted potency,
not a binarized hit call (unlike the TDC/MoleculeNet repackaging of this same assay).

Only the "exact" (curve-fitted) arm is pulled -- inactive/censored rows are dropped, not
turned into bounds, to keep this a first, simple version.
"""

import io
import logging
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("pull_pubchem")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external"

AID = 1851
REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
SID_CHUNK = 8000

ISOFORMS = {
    "p450-cyp1a2": "CYP1A2",
    "p450-cyp2c9": "CYP2C9",
    "p450-cyp2d6": "CYP2D6",
    "p450-cyp3a4": "CYP3A4",
}

COLUMNS = {
    "PUBCHEM_SID": "sid",
    "PUBCHEM_CID": "cid",
    "PUBCHEM_EXT_DATASOURCE_SMILES": "orig_smiles",
    "PUBCHEM_ACTIVITY_OUTCOME": "activity_outcome",
    "Panel Name": "isoform_slug",
    "Fit_LogAC50": "fit_log_ac50",
    "Fit_CurveClass": "curve_class",
    "Fit_R2": "fit_r2",
}


def fetch_sids() -> list[str]:
    resp = requests.get(f"{REST}/assay/aid/{AID}/sids/TXT", timeout=120)
    resp.raise_for_status()
    sids = resp.text.split()
    log.info(f"AID {AID}: {len(sids):,} SIDs")
    return sids


def fetch_chunk(sids: list[str]) -> pd.DataFrame:
    resp = requests.post(f"{REST}/assay/aid/{AID}/CSV", data={"sid": ",".join(sids)}, timeout=600)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text), low_memory=False)


def standardize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    from rdkit.Chem.MolStandardize import rdMolStandardize
    frag_chooser = rdMolStandardize.LargestFragmentChooser()
    mol = frag_chooser.choose(mol)
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sids = fetch_sids()
    chunks = [sids[i : i + SID_CHUNK] for i in range(0, len(sids), SID_CHUNK)]
    frames = []
    for n, chunk in enumerate(chunks, start=1):
        log.info(f"Fetching chunk {n}/{len(chunks)} ({len(chunk):,} SIDs)")
        frames.append(fetch_chunk(chunk))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["PUBCHEM_SID"].notna()]

    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    df["isoform"] = df["isoform_slug"].map(ISOFORMS)
    df = df[df["isoform"].notna()].drop(columns=["isoform_slug"])
    df["fit_log_ac50"] = pd.to_numeric(df["fit_log_ac50"], errors="coerce")
    df["pic50"] = -df["fit_log_ac50"]
    df = df.dropna(subset=["pic50"])  # exact-fit arm only, no censored bounds yet

    log.info(f"{len(df):,} rows with a fitted pIC50 across {df['sid'].nunique():,} SIDs")

    log.info("Standardizing SMILES...")
    unique_smiles = df["orig_smiles"].dropna().unique()
    smiles_map = {s: standardize_smiles(s) for s in unique_smiles}
    df["smiles"] = df["orig_smiles"].map(smiles_map)
    dropped = df["smiles"].isna().sum()
    if dropped:
        log.warning(f"Dropping {dropped:,} rows whose SMILES could not be standardized")
    df = df.dropna(subset=["smiles"])

    out_path = OUTPUT_DIR / "pubchem_aid1851_cyp_raw.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Wrote {len(df):,} rows -> {out_path}")
    for iso in ISOFORMS.values():
        n = df[df["isoform"] == iso]["smiles"].nunique()
        log.info(f"  {iso}: {n:,} unique compounds with fitted pIC50")


if __name__ == "__main__":
    main()
