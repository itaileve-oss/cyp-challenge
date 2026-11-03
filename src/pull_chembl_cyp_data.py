"""Pull CYP inhibition potency (pchembl_value) from ChEMBL for our 4 isoforms.

Adapted from SuperCowPowers/workbench's pull_chembl_cyp_data.py, which reads a private S3
parquet mirror of the full release -- we don't have that bucket, so this hits the public
ChEMBL REST API directly (chembl_webresource_client's own pagination turned out to be far
slower than the raw API -- likely a small default page size -- so this uses limit=1000
pages directly, ~8s/page, ~9 min total for all 4 targets' ~67.5k activities).

Only pchembl_value-bearing "exact" activities are pulled (a curve was actually fitted) --
same simple-first scope as the PubChem pull. `confidence_score` filtering is done
post-hoc since the REST API's `activity` endpoint doesn't expose the assay's confidence
score directly as a filter; we go through `assay_chembl_id` -> `confidence_score` via a
second bulk lookup instead of filtering per-activity.

Writes one CSV per target immediately after it finishes (not just at the very end), so an
interrupted run doesn't lose completed targets.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger("pull_chembl")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s", force=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external"

API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000

TARGETS = {
    "CHEMBL3356": "CYP1A2",
    "CHEMBL3397": "CYP2C9",
    "CHEMBL289": "CYP2D6",
    "CHEMBL340": "CYP3A4",
}

FIELDS = [
    "molecule_chembl_id", "canonical_smiles", "pchembl_value",
    "data_validity_comment", "potential_duplicate", "assay_chembl_id",
]


def fetch_target_activities(chembl_target_id: str, isoform: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        params = {
            "target_chembl_id": chembl_target_id,
            "pchembl_value__isnull": "false",
            "target_organism": "Homo sapiens",
            "limit": PAGE_SIZE,
            "offset": offset,
            "only": ",".join(FIELDS),
        }
        t0 = time.time()
        resp = requests.get(f"{API}/activity.json", params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        batch = data["activities"]
        rows.extend(batch)
        total = data["page_meta"]["total_count"]
        log.info(f"  {isoform}: {len(rows):,}/{total:,} ({time.time()-t0:.1f}s/page)")
        offset += PAGE_SIZE
        if offset >= total or not batch:
            break
    df = pd.DataFrame(rows)
    df["isoform"] = isoform
    return df


def filter_by_confidence(df: pd.DataFrame, min_confidence: int = 8) -> pd.DataFrame:
    """Look up each referenced assay's confidence_score in bulk and filter."""
    assay_ids = df["assay_chembl_id"].dropna().unique().tolist()
    log.info(f"  Looking up confidence_score for {len(assay_ids):,} assays...")
    confidence = {}
    ASSAY_CHUNK = 50  # assay_chembl_id__in goes in the URL query string -- keep it short
    for i in range(0, len(assay_ids), ASSAY_CHUNK):
        chunk = assay_ids[i : i + ASSAY_CHUNK]
        params = {
            "assay_chembl_id__in": ",".join(chunk),
            "limit": ASSAY_CHUNK,
            "only": "assay_chembl_id,confidence_score",
        }
        resp = requests.get(f"{API}/assay.json", params=params, timeout=120)
        resp.raise_for_status()
        for a in resp.json()["assays"]:
            confidence[a["assay_chembl_id"]] = a["confidence_score"]
    df["confidence_score"] = df["assay_chembl_id"].map(confidence)
    before = len(df)
    df = df[pd.to_numeric(df["confidence_score"], errors="coerce") >= min_confidence]
    log.info(f"  confidence >= {min_confidence}: kept {len(df):,}/{before:,}")
    return df


def standardize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    from rdkit.Chem.MolStandardize import rdMolStandardize
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_target_paths = []

    for tid, iso in TARGETS.items():
        out_path = OUTPUT_DIR / f"chembl_{iso.lower()}_raw.csv"
        if out_path.exists():
            log.info(f"{iso}: already pulled -> {out_path} (skipping fetch)")
            per_target_paths.append(out_path)
            continue

        log.info(f"=== {iso} ({tid}) ===")
        raw_cache_path = OUTPUT_DIR / f"chembl_{iso.lower()}_raw_unfiltered.csv"
        if raw_cache_path.exists():
            log.info(f"{iso}: reusing cached raw fetch -> {raw_cache_path}")
            df = pd.read_csv(raw_cache_path)
        else:
            df = fetch_target_activities(tid, iso)
            df.to_csv(raw_cache_path, index=False)
            log.info(f"{iso}: cached {len(df):,} raw rows -> {raw_cache_path}")
        df = df[df["data_validity_comment"].isna()]
        df = df[df["potential_duplicate"].isin([0, "0", None]) | df["potential_duplicate"].isna()]
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.dropna(subset=["pchembl_value", "canonical_smiles"])
        df = filter_by_confidence(df)

        df = df.rename(columns={"pchembl_value": "pic50", "canonical_smiles": "orig_smiles"})
        df = df[["molecule_chembl_id", "orig_smiles", "pic50", "isoform"]]
        df.to_csv(out_path, index=False)
        log.info(f"{iso}: wrote {len(df):,} rows -> {out_path}")
        per_target_paths.append(out_path)

    log.info("Standardizing SMILES across all targets...")
    all_df = pd.concat([pd.read_csv(p) for p in per_target_paths], ignore_index=True)
    unique_smiles = all_df["orig_smiles"].dropna().unique()
    log.info(f"  {len(unique_smiles):,} unique raw SMILES to standardize")
    smiles_map = {s: standardize_smiles(s) for s in unique_smiles}
    all_df["smiles"] = all_df["orig_smiles"].map(smiles_map)
    dropped = all_df["smiles"].isna().sum()
    if dropped:
        log.warning(f"Dropping {dropped:,} rows whose SMILES could not be standardized")
    all_df = all_df.dropna(subset=["smiles"])

    combined_path = OUTPUT_DIR / "chembl_cyp_raw.csv"
    all_df.to_csv(combined_path, index=False)
    log.info(f"Wrote {len(all_df):,} combined rows -> {combined_path}")
    for iso in TARGETS.values():
        n = all_df[all_df["isoform"] == iso]["smiles"].nunique()
        log.info(f"  {iso}: {n:,} unique compounds with pchembl_value (confidence-filtered)")


if __name__ == "__main__":
    main()
