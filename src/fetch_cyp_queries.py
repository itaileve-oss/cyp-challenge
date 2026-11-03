"""Fetch and filter real bound-ligand query structures for a CYP isoform from the PDB.

Terminology (ROCS convention): "query" = a molecule with a known 3D pose we align
onto (real PDB-bound CYP inhibitor structures, fetched here); "fit" = a molecule we
search for a pose for by superposing it onto a query (our own SMILES, no
experimental structure -- see prepare_fit_ligands.py).

Step 1 of the pose-alignment pipeline: given a UniProt accession, find all PDB
structures, enumerate their non-polymer (heteroatom) components, and filter down to
genuine drug-like bound ligands -- excluding the heme cofactor (handled separately),
crystallization additives, and non-heme-metal probes (this CYP3A4 set includes
several Ir/Ru bipyridine photocatalyst structures, not real inhibitors).

Does NOT yet extract 3D coordinates or fix bond orders -- that's
prepare_query_structures.py, once we have the filtered (pdbid, comp_id) list.
"""

import json
import re
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIPROT_ACCESSIONS = {
    "CYP3A4": "P08684",
    "CYP2D6": "P10635",
    "CYP2C9": "P11712",
    "CYP1A2": "P05177",
}

# Crystallization additives / cryoprotectants / buffer ions seen in practice, plus
# common metal ions. HEM (the cofactor) is excluded separately, not in this list,
# since it's meaningful (defines the reactive site) even though it's not a query
# ligand itself.
ADDITIVE_BLOCKLIST = {
    "HOH", "WAT", "GOL", "EDO", "SO4", "PO4", "DMS", "PG4", "PG0", "PGE", "ACT",
    "PMF", "IMD", "1PE", "MPD", "TRS", "BME", "EPE", "FMT", "CIT", "TLA",
    "NA", "CL", "MG", "CA", "ZN", "MN", "K", "CO", "NI", "CU", "FE", "CD",
}
MIN_HEAVY_ATOMS = 6


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def get_pdb_ids(uniprot_accession: str) -> list[str]:
    data = fetch_json(f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}.json")
    return [x["id"] for x in data.get("uniProtKBCrossReferences", []) if x["database"] == "PDB"]


def get_ligand_inventory(pdb_ids: list[str]) -> list[tuple[str, str, str, str]]:
    """Returns (pdb_id, comp_id, name, formula) for every non-polymer entity across
    all given PDB entries.
    """
    rows = []
    for i, pdbid in enumerate(pdb_ids):
        try:
            entry = fetch_json(f"https://data.rcsb.org/rest/v1/core/entry/{pdbid}")
            eids = entry.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", []) or []
            for eid in eids:
                ent = fetch_json(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdbid}/{eid}")
                comp = ent.get("pdbx_entity_nonpoly", {})
                comp_id = comp.get("comp_id", "")
                name = comp.get("name", "")
                formula = ent.get("rcsb_nonpolymer_entity", {}).get("formula_weight")
                rows.append((pdbid, comp_id, name, str(formula)))
        except Exception as e:
            rows.append((pdbid, "ERROR", str(e), ""))
        if (i + 1) % 25 == 0:
            print(f"  scanned {i + 1}/{len(pdb_ids)} structures", flush=True)
    return rows


def get_component_formula_info(comp_id: str) -> tuple[set[str], int]:
    """(element symbols, heavy-atom count) for a chemical component, from its
    molecular formula (e.g. "C37 H48 N6 O5 S2" -> {C,H,N,O,S}, 50 heavy atoms).
    """
    try:
        data = fetch_json(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}")
        formula = data.get("chem_comp", {}).get("formula", "")
    except Exception:
        return set(), 0
    elements = set()
    n_heavy = 0
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not symbol:
            continue
        elements.add(symbol)
        n = int(count) if count else 1
        if symbol != "H":
            n_heavy += n
    return elements, n_heavy


NON_HEME_TRANSITION_METALS = {"Ir", "Ru", "Rh", "Pd", "Pt", "Os", "Re", "W", "Mo"}


def filter_queries(rows: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str]]:
    """Apply the blocklist + heavy-atom floor + non-heme-metal exclusion. Returns
    deduplicated (pdb_id, comp_id, name) tuples for real candidate query ligands.
    """
    seen_comp_ids: dict[str, tuple[set[str], int]] = {}  # comp_id -> (elements, n_heavy), cached
    filtered = []
    for pdbid, comp_id, name, _formula in rows:
        if comp_id in ("HEM", "ERROR", "") or comp_id in ADDITIVE_BLOCKLIST:
            continue
        if comp_id not in seen_comp_ids:
            seen_comp_ids[comp_id] = get_component_formula_info(comp_id)
            time.sleep(0.02)  # be polite to the API
        elements, n_heavy = seen_comp_ids[comp_id]
        if elements & NON_HEME_TRANSITION_METALS:
            continue
        if n_heavy < MIN_HEAVY_ATOMS:
            continue
        filtered.append((pdbid, comp_id, name))
    return filtered


def main() -> None:
    import sys
    isoform = sys.argv[1] if len(sys.argv) > 1 else "CYP3A4"
    accession = UNIPROT_ACCESSIONS[isoform]
    print(f"Fetching PDB IDs for {isoform} ({accession})...", flush=True)
    pdb_ids = get_pdb_ids(accession)
    print(f"  {len(pdb_ids)} structures", flush=True)

    print("Scanning non-polymer components...", flush=True)
    rows = get_ligand_inventory(pdb_ids)

    out_raw = PROJECT_ROOT / "data" / "structures" / f"{isoform.lower()}_ligand_inventory_raw.tsv"
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    with open(out_raw, "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"Raw inventory saved to {out_raw} ({len(rows)} rows)")

    print("Filtering to candidate query ligands...", flush=True)
    filtered = filter_queries(rows)
    out_filtered = PROJECT_ROOT / "data" / "structures" / f"{isoform.lower()}_queries_filtered.tsv"
    with open(out_filtered, "w") as f:
        for r in filtered:
            f.write("\t".join(r) + "\n")
    print(f"Filtered queries saved to {out_filtered} ({len(filtered)} rows, "
          f"{len(set(r[1] for r in filtered))} unique compounds)")


if __name__ == "__main__":
    main()
