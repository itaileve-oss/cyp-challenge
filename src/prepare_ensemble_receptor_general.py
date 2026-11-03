"""General receptor prep for any CYP isoform/PDB entry: strip everything except
protein + HEM, pdbfixer (missing residues/atoms + H at pH 7.4), compute the box
center from the co-crystallized reference ligand's heavy-atom centroid (BEFORE
stripping it out), and report the proximal Cys (heme-ligating thiolate) residue.
Same pattern as prepare_cyp3a4_receptor.py, generalized to not hardcode the
ligand resname or input/output paths.

Usage: python prepare_ensemble_receptor_general.py <input.pdb> <output.pdb> <ref_ligand_resname> [chain]

Optional [chain]: restrict to a single chain before running pdbfixer. Several
of the CYP2D6 structures have 4 protein copies per asymmetric unit (~15,000+
ATOM/HETATM lines vs ~8,000 for a single copy), which made pdbfixer's missing-
residue/atom reconstruction take 45+ minutes per structure with no sign of
finishing -- restricting to one chain (with its own HEM/ligand) first avoids
redoing that reconstruction 4x over.
"""

import sys
from pathlib import Path

import numpy as np
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import openmm.app as app

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def compute_ref_ligand_centroid(pdb_path: Path, ref_resname: str) -> np.ndarray | None:
    coords = []
    for line in pdb_path.read_text().splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == ref_resname:
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        return None
    return np.array(coords).mean(axis=0)


def restrict_to_chain(input_pdb: Path, chain: str) -> Path:
    lines = input_pdb.read_text().splitlines()
    keep = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            if line[21] == chain:
                keep.append(line)
        elif line.startswith(("TER", "END")):
            keep.append(line)
    out_path = input_pdb.with_name(input_pdb.stem + f"_chain{chain}.pdb")
    out_path.write_text("\n".join(keep) + "\n")
    return out_path


def main():
    input_pdb = Path(sys.argv[1])
    output_pdb = Path(sys.argv[2])
    ref_resname = sys.argv[3]
    chain = sys.argv[4] if len(sys.argv) > 4 else None

    if chain:
        print(f"Restricting to chain {chain} before pdbfixer...")
        input_pdb = restrict_to_chain(input_pdb, chain)

    centroid = compute_ref_ligand_centroid(input_pdb, ref_resname)
    if centroid is None:
        print(f"WARNING: reference ligand {ref_resname} not found in {input_pdb}, no box center computed")
    else:
        print(f"Box center (from {ref_resname} centroid): {centroid[0]:.3f} {centroid[1]:.3f} {centroid[2]:.3f}")

    fixer = PDBFixer(filename=str(input_pdb))
    modeller = app.Modeller(fixer.topology, fixer.positions)
    to_delete = [
        res for res in modeller.topology.residues()
        if res.name not in STANDARD_AA and res.name != "HEM"
    ]
    modeller.delete(to_delete)
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    print(f"Prepared receptor written to: {output_pdb}")

    # verify HEM present, find Fe + proximal Cys
    fe = None
    lines = output_pdb.read_text().splitlines()
    for line in lines:
        if line.startswith("HETATM") and line[76:78].strip() == "Fe":
            fe = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if fe is None:
        print("ERROR: no Fe atom found in prepared receptor!")
        return
    print(f"Fe coord: {fe}")

    cys_sg = []
    for line in lines:
        if line.startswith("ATOM") and line[17:20].strip() == "CYS" and line[12:16].strip() == "SG":
            resnum = line[22:26].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            cys_sg.append((resnum, float(np.linalg.norm(xyz - fe))))
    cys_sg.sort(key=lambda t: t[1])
    if cys_sg:
        print(f"Closest CYS to Fe: CYS{cys_sg[0][0]} at {cys_sg[0][1]:.2f} A "
              f"(next closest: {cys_sg[1][1]:.2f} A)" if len(cys_sg) > 1 else "")

    if centroid is not None:
        box_path = output_pdb.with_suffix(".box.txt")
        box_path.write_text(f"{centroid[0]:.3f} {centroid[1]:.3f} {centroid[2]:.3f}\n")
        print(f"Box center saved to {box_path}")


if __name__ == "__main__":
    main()
