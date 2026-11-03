"""Extract and prepare 3D query structures (real PDB-bound CYP inhibitor poses).

Terminology: see prepare_fit_ligands.py -- "query" = known 3D pose we align onto.

For each (pdb_id, comp_id) pair from fetch_cyp_queries.py's filtered list:
  1. Download the mmCIF structure (cached) and parse it with gemmi -- NOT the
     legacy fixed-column .pdb format, which silently fails to even exist for newer
     PDB entries once a component's comp_id exceeds the legacy format's 3-character
     column width (RCSB introduced 5-character IDs for new ligands at some point;
     19/101 of our CYP3A4 query candidates use them, e.g. "A1APA" = cobicistat).
     gemmi has no such limitation.
  2. Extract the ligand's atom coordinates -- correct experimental 3D geometry, but
     WRONG/no bond order info (X-ray geometry alone doesn't encode bond order or
     aromaticity).
  3. Fetch the component's reference SMILES from RCSB (correct bond orders/
     aromaticity), protonate it at pH 7.4 (same dimorphite_dl treatment as fit
     molecules, for consistency), and use RDKit's AssignBondOrdersFromTemplate to
     transfer correct bonding onto the raw crystal-derived mol -- WITHOUT touching
     the experimental heavy-atom coordinates. That coordinate is the actual
     bioactive-pose signal; never re-embed or re-optimize it.
  4. Add explicit hydrogens with RDKit-placed coordinates (addCoords=True) --
     reasonable local geometry, not energy-minimized, but X-ray doesn't resolve
     ligand hydrogens anyway so this is the best available.

Run in the `GNINA` conda env (has dimorphite_dl, gemmi).
"""

import json
import pickle
import time
import urllib.request
from pathlib import Path

import dimorphite_dl
import gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDetermineBonds

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CIF_CACHE = PROJECT_ROOT / "data" / "structures" / "raw_cif"
RAW_CIF_CACHE.mkdir(parents=True, exist_ok=True)

DEFAULT_PH = 7.4


def fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def fetch_structure(pdbid: str) -> gemmi.Structure:
    cache_path = RAW_CIF_CACHE / f"{pdbid}.cif"
    if not cache_path.exists():
        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdbid}.cif", timeout=30) as r:
            cache_path.write_bytes(r.read())
    return gemmi.read_structure(str(cache_path))


def get_reference_smiles(comp_id: str) -> str | None:
    """Prefer the CACTVS canonical SMILES (has stereo); fall back to any SMILES
    descriptor available for the component.
    """
    try:
        data = fetch_json(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}")
    except Exception:
        return None
    descriptors = data.get("pdbx_chem_comp_descriptor", [])
    smiles_entries = [d for d in descriptors if d.get("type", "").startswith("SMILES")]
    if not smiles_entries:
        return None
    cactvs = [d for d in smiles_entries if d.get("program") == "CACTVS" and d["type"] == "SMILES_CANONICAL"]
    chosen = cactvs[0] if cactvs else smiles_entries[0]
    return chosen.get("descriptor")


def extract_ligand_raw_mol(structure: gemmi.Structure, comp_id: str) -> Chem.Mol | None:
    """Build a heavy-atom-only RDKit mol (real crystallographic 3D coordinates,
    distance-perceived connectivity, no bond orders yet) from the first instance of
    comp_id found in the structure's first model.
    """
    model = structure[0]
    target_residue = None
    for chain in model:
        for residue in chain:
            if residue.name == comp_id:
                target_residue = residue
                break
        if target_residue is not None:
            break
    if target_residue is None:
        return None

    mol = Chem.RWMol()
    conf = Chem.Conformer()
    n_atoms = 0
    seen_altloc_primary = None
    for atom in target_residue:
        # Keep only the primary/highest-occupancy alternate location, matching the
        # legacy-format approach (blank altloc, or 'A' if no blank present).
        if atom.altloc not in ("\0", "A"):
            continue
        element_symbol = atom.element.name
        if element_symbol == "H":
            continue  # X-ray doesn't resolve ligand Hs; RDKit adds them back later
        rd_atom = Chem.Atom(element_symbol)
        mol.AddAtom(rd_atom)
        pos = atom.pos
        conf.SetAtomPosition(n_atoms, (pos.x, pos.y, pos.z))
        n_atoms += 1

    if n_atoms == 0:
        return None
    conf.SetId(0)
    mol.AddConformer(conf, assignId=True)
    mol = mol.GetMol()
    try:
        rdDetermineBonds.DetermineConnectivity(mol)
    except Exception:
        return None
    return mol


def protonate_dominant(smiles: str, ph: float = DEFAULT_PH) -> str | None:
    try:
        variants = dimorphite_dl.protonate_smiles(smiles, ph_min=ph, ph_max=ph)
    except Exception:
        return None
    return variants[0] if variants else None


def build_query_mol(pdbid: str, comp_id: str) -> Chem.Mol | None:
    ref_smiles = get_reference_smiles(comp_id)
    if ref_smiles is None:
        return None
    protonated = protonate_dominant(ref_smiles)
    template_smiles = protonated if protonated is not None else ref_smiles
    template_mol = Chem.MolFromSmiles(template_smiles)
    if template_mol is None:
        template_mol = Chem.MolFromSmiles(ref_smiles)
        if template_mol is None:
            return None

    try:
        structure = fetch_structure(pdbid)
    except Exception:
        return None
    raw_mol = extract_ligand_raw_mol(structure, comp_id)
    if raw_mol is None or raw_mol.GetNumAtoms() == 0:
        return None

    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template_mol, raw_mol)
    except Exception:
        return None

    # DetermineConnectivity marks every atom noImplicit=True (it assumes only
    # explicitly-placed atoms should count, no auto-filled valence). With that flag
    # still set, AssignBondOrdersFromTemplate satisfies any leftover valence with
    # RADICAL ELECTRONS instead of leaving room for implicit Hs (verified: e.g. a
    # 2-bonded carbon came out "valence 2, 0 H, 2 radical electrons" instead of the
    # correct "valence 4, 2 H" -- silently produced a mol with roughly half the
    # correct heavy+H atom count once AddHs ran). Clear both flags and refresh
    # valence bookkeeping before sanitizing so implicit-H calculation (and thus
    # AddHs) works correctly.
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    try:
        mol = Chem.AddHs(mol, addCoords=True)
    except Exception:
        pass  # keep heavy-atom-only mol rather than losing it entirely

    mol.SetProp("_Name", f"{pdbid}_{comp_id}")
    return mol


def main() -> None:
    isoform = "cyp3a4"
    filtered_path = PROJECT_ROOT / "data" / "structures" / f"{isoform}_queries_filtered.tsv"
    rows = [line.split("\t") for line in filtered_path.read_text().splitlines() if line.strip()]
    print(f"{len(rows)} (pdb_id, comp_id) pairs to process", flush=True)

    query_mols: dict[tuple[str, str], Chem.Mol] = {}
    failed = []
    start = time.time()
    for i, (pdbid, comp_id, _name) in enumerate(rows):
        mol = build_query_mol(pdbid, comp_id)
        if mol is None:
            failed.append((pdbid, comp_id))
        else:
            query_mols[(pdbid, comp_id)] = mol
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)} ({len(query_mols)} ok, {len(failed)} failed)", flush=True)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s. {len(query_mols)}/{len(rows)} succeeded.")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:20]}{'...' if len(failed) > 20 else ''}")

    out_path = PROJECT_ROOT / "data" / "structures" / f"{isoform}_query_mols.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(query_mols, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
