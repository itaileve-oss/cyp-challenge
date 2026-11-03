"""Write one Boltz-2 YAML per structure-track ligand.

Each complex is protein (CYP3A4) + heme (CCD HEM) + ligand (SMILES from the
blinded CSV, used verbatim so the submission validator's
AssignBondOrdersFromTemplate check cannot fail on connectivity).

The axial Cys442-SG--heme-FE bond is the conserved P450 proximal ligand; passing
it as a Boltz bond constraint anchors the cofactor instead of leaving the model
to rediscover it. Optional because it is the main thing worth A/B-ing in
calibration.

Usage:
    python make_boltz_inputs.py OUT_DIR [--bond] [--msa PATH.a3m]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FASTA = ROOT / "external/CYP-Challenge-Tutorial/inputs/CYP3A4_protein_sequence.fasta"
LIGANDS = ROOT / "data/raw/TEST_BLINDED_structures.csv"

AXIAL_CYS = 442  # verified against P08684: ...YTPFGSGPRNC(442)IGMR...


def read_sequence() -> str:
    lines = FASTA.read_text().strip().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def build_yaml(sequence: str, smiles: str, msa: str | None, bond: bool) -> str:
    protein = f"- protein: {{id: A, sequence: {sequence}"
    protein += f", msa: {msa}}}" if msa else "}"

    blocks = [
        "version: 1",
        "sequences:",
        protein,
        "- ligand: {id: B, ccd: HEM}",
        f"- ligand: {{id: C, smiles: '{smiles}'}}",
    ]
    if bond:
        blocks += [
            "constraints:",
            "- bond:",
            f"    atom1: [A, {AXIAL_CYS}, SG]",
            "    atom2: [B, 1, FE]",
        ]
    return "\n".join(blocks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--bond", action="store_true", help="add Cys442-SG--FE bond constraint")
    ap.add_argument("--msa", help="precomputed a3m to reuse (skips the MSA server per ligand)")
    ap.add_argument("--only", help="write just this Molecule_Name (for calibration)")
    args = ap.parse_args()

    sequence = read_sequence()
    if sequence[AXIAL_CYS - 1] != "C":
        raise SystemExit(f"position {AXIAL_CYS} is {sequence[AXIAL_CYS - 1]!r}, expected C")

    ligands = pd.read_csv(LIGANDS)
    if args.only:
        ligands = ligands[ligands["Molecule_Name"] == args.only]
        if ligands.empty:
            raise SystemExit(f"no ligand named {args.only}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, smiles in zip(ligands["Molecule_Name"], ligands["SMILES"]):
        path = args.out_dir / f"{name}.yaml"
        path.write_text(build_yaml(sequence, smiles, args.msa, args.bond))
    print(f"wrote {len(ligands)} YAML(s) to {args.out_dir} (bond={args.bond}, msa={args.msa})")


if __name__ == "__main__":
    main()
