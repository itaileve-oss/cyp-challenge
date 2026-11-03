"""Assemble and validate a CYP structure-track submission zip.

One `{Molecule_Name}.pdb` per ligand, taken from the selected pose. The pose
PDBs written by pb_official.py already carry the heme renamed to HEM, which the
submission requires: Boltz names both the cofactor and the ligand LIG, and the
validator demands exactly one residue named LIG.

Validation is the organizers' own `validate_structure_submission`, run with the
expected IDs and the challenge SMILES so ligand connectivity is checked the way
it will be at scoring time. Nothing is uploaded - this only produces the zip.

Usage:
    python build_submission.py SELECTED_CSV OUT_ZIP
"""
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/CYP-Challenge-Tutorial"))
from validation.structure_validation import validate_structure_submission  # noqa: E402


def main(selected_csv: str, out_zip: str) -> None:
    sel = pd.read_csv(selected_csv)
    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)

    staging = out.parent / f"{out.stem}_files"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    missing = sorted(set(smi.index) - set(sel["Molecule_Name"]))
    if missing:
        print(f"WARNING {len(missing)} ligand(s) absent from the selection: {missing}")

    for _, r in sel.iterrows():
        src = Path(r["sub_pdb"])
        if not src.exists():
            print(f"WARNING missing pose file for {r['Molecule_Name']}: {src}")
            continue
        shutil.copy(src, staging / f"{r['Molecule_Name']}.pdb")

    pdbs = sorted(staging.glob("*.pdb"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pdbs:
            zf.write(p, p.name)
    print(f"wrote {out} with {len(pdbs)} structures "
          f"({out.stat().st_size / 1e6:.1f} MB)")

    ok, errors = validate_structure_submission(
        out, expected_ids=set(smi.index),
        expected_ligand_smiles=smi.to_dict(),
    )
    print(f"\nofficial validator: {'PASS' if ok else 'FAIL'}")
    for e in errors:
        print(f"  - {e}")
    if ok:
        print("Submission is valid and ready to upload (not uploaded).")


if __name__ == "__main__":
    main(*sys.argv[1:])
