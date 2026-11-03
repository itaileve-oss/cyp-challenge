"""Prepare the CYP3A4 receptor (PDB 4NY4) for docking/scoring, following the same
pdbfixer pattern validated in the OpenBind/GNINA project
(~/Projects/GNINA_test/prepare_protein.py):
- Strip the bound ligand (2QH) and any crystallization solvent, keep HEM (cofactor)
- Fix missing residues/heavy atoms
- Add hydrogens at physiological pH (7.4)

Replaces the pre-existing data/receptors/cyp3a4_receptor.pdb, which turned out to
have 0 explicit hydrogens and no pdbfixer pass -- just a HETATM strip, found while
auditing prep quality (see project memory, 2026-09-12).

Run: conda run -n GNINA python src/prepare_cyp3a4_receptor.py
"""

from pathlib import Path

from pdbfixer import PDBFixer
from openmm.app import PDBFile
import openmm.app as app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "data" / "receptors" / "4ny4.pdb"
OUTPUT = PROJECT_ROOT / "data" / "receptors" / "cyp3a4_receptor_prepared.pdb"

REMOVE_RESNAMES = {"2QH", "HOH", "WAT", "DMS", "GOL", "EDO", "SO4", "PO4"}

fixer = PDBFixer(filename=str(INPUT))

modeller = app.Modeller(fixer.topology, fixer.positions)
to_delete = [res for res in modeller.topology.residues() if res.name in REMOVE_RESNAMES]
modeller.delete(to_delete)
fixer.topology = modeller.topology
fixer.positions = modeller.positions

fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()

fixer.addMissingHydrogens(pH=7.4)

with open(OUTPUT, "w") as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

print(f"Prepared receptor written to: {OUTPUT}")
counts = {}
with open(OUTPUT) as f:
    for line in f:
        if line.startswith(("ATOM", "HETATM")):
            resname = line[17:20].strip()
            counts[resname] = counts.get(resname, 0) + 1

standard_aa = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
print("Non-standard residues retained:")
for k, v in sorted(counts.items()):
    if k not in standard_aa:
        print(f"  {k}: {v} atoms")
print(f"Total ATOM/HETATM lines: {sum(counts.values())}")
