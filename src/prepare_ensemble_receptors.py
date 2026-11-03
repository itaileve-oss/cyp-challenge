"""Prepare 4 additional CYP3A4 receptor conformations for ensemble docking,
each superimposed into the same coordinate frame as the existing 4NY4 receptor
(so the same GalaxyDock2-HEME box center works for all of them).

Chosen to span CYP3A4's known induced-fit pocket range:
- 1W0F (progesterone, STR)  -- small steroid, compact pocket
- 1W0G (metyrapone, MYT)    -- small pyridine, classic heme-N-ligating inhibitor
- 3NXU (ritonavir, RIT)     -- large peptidomimetic, known to massively expand the pocket
- 2J0D (erythromycin A, ERY)-- large macrolide, different expansion mode

Same superposition (gemmi Calpha rigid-body fit onto 4NY4) as
align_queries_to_receptor.py, but applied to the WHOLE structure (not just the
bound ligand), then run through the same pdbfixer pipeline as
prepare_cyp3a4_receptor.py (strip ligand/solvent, keep HEM, fix missing atoms,
add hydrogens at pH 7.4).
"""

from pathlib import Path

import gemmi
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import openmm.app as app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CIF_CACHE = PROJECT_ROOT / "data" / "structures" / "raw_cif"
OUT_DIR = PROJECT_ROOT / "data" / "receptors" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_PDBID = "4NY4"

TARGETS = {
    "1W0F": "STR",
    "1W0G": "MYT",
    "3NXU": "RIT",
    "2J0D": "ERY",
}

REMOVE_RESNAMES_BASE = {"HOH", "WAT", "DMS", "GOL", "EDO", "SO4", "PO4", "ACT", "TRS", "PEG", "MPD", "EPE"}


def load_structure(pdbid: str) -> gemmi.Structure:
    return gemmi.read_structure(str(RAW_CIF_CACHE / f"{pdbid}.cif"))


def find_ligand_chain(structure: gemmi.Structure, comp_id: str) -> str | None:
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.name == comp_id:
                return chain.name
    return None


def get_polymer_chain(structure: gemmi.Structure, preferred_chain_name: str | None):
    model = structure[0]
    if preferred_chain_name is not None:
        for chain in model:
            if chain.name == preferred_chain_name:
                polymer = chain.get_polymer()
                if len(polymer) > 0:
                    return polymer
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) > 0:
            return polymer
    return None


def find_fe_coord(pdb_path: Path):
    for line in pdb_path.read_text().splitlines():
        if line.startswith("HETATM") and line[76:78].strip() == "Fe":
            return float(line[30:38]), float(line[38:46]), float(line[46:54])
    return None


def main():
    ref_structure = load_structure(REFERENCE_PDBID)
    ref_polymer = get_polymer_chain(ref_structure, preferred_chain_name="A")
    print(f"Reference: {REFERENCE_PDBID}, {len(ref_polymer)} polymer residues", flush=True)

    for pdbid, lig_code in TARGETS.items():
        print(f"\n=== {pdbid} ({lig_code}) ===", flush=True)
        structure = load_structure(pdbid)
        ligand_chain_name = find_ligand_chain(structure, lig_code)
        movable_polymer = get_polymer_chain(structure, preferred_chain_name=ligand_chain_name)
        if movable_polymer is None:
            print(f"  SKIP: no polymer chain found")
            continue

        result = gemmi.calculate_superposition(
            ref_polymer, movable_polymer, gemmi.PolymerType.PeptideL, gemmi.SupSelect.CaP
        )
        transform = result.transform
        print(f"  Calpha superposition RMSD onto {REFERENCE_PDBID}: {result.rmsd:.2f} A")

        # Apply the transform to every atom in the whole structure.
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        pos = atom.pos
                        new_pos = transform.apply(gemmi.Position(pos.x, pos.y, pos.z))
                        atom.pos = gemmi.Position(new_pos.x, new_pos.y, new_pos.z)

        raw_aligned_pdb = OUT_DIR / f"{pdbid.lower()}_aligned_raw.pdb"
        structure.setup_entities()
        structure.write_pdb(str(raw_aligned_pdb))

        # Strip bound ligand + crystallization solvent, keep HEM, via the same
        # pdbfixer/Modeller pipeline as prepare_cyp3a4_receptor.py.
        remove_resnames = REMOVE_RESNAMES_BASE | {lig_code}
        fixer = PDBFixer(filename=str(raw_aligned_pdb))
        modeller = app.Modeller(fixer.topology, fixer.positions)
        to_delete = [res for res in modeller.topology.residues() if res.name in remove_resnames]
        modeller.delete(to_delete)
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.4)

        out_pdb = OUT_DIR / f"cyp3a4_{pdbid.lower()}_prepared.pdb"
        with open(out_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

        fe = find_fe_coord(out_pdb)
        print(f"  Prepared receptor: {out_pdb}")
        print(f"  Fe coordinate: {fe}")

        heme_chain = None
        heme_resnum = None
        for line in out_pdb.read_text().splitlines():
            if line.startswith("HETATM") and line[17:20].strip() == "HEM":
                heme_chain = line[21]
                heme_resnum = line[22:26].strip()
                break
        print(f"  HEM residue: chain={heme_chain}, res_num={heme_resnum}")


if __name__ == "__main__":
    main()
