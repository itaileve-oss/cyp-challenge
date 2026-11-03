"""Run the organizers' exact scoring-time PoseBusters check on pool poses.

Two differences from a naive `PoseBusters().bust()` call matter, so both are
reproduced here rather than approximated:

1. The submission renames the heme from LIG to HEM (Boltz names the cofactor
   and the ligand alike), so at scoring time the heme sits in `pb_protein`
   instead of being popped out with the ligand.
2. `evaluate_predictions.score_single_structure` passes the challenge SMILES as
   `mol_true`, which enables checks a `mol_true=None` call never runs.

Writes submission-style PDBs as a side effect; `build_submission.py` reuses them.

Usage:
    python pb_official.py OUT_DIR PRED_DIR [PRED_DIR ...]
"""
import sys
import warnings
from pathlib import Path

import gemmi
import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/CYP-Challenge-Tutorial"))
from evaluation.posebusters_utils import (  # noqa: E402
    assign_bond_orders_from_template,
    dedupe_ligand_copies,
)

SMILES = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def pose_id(src: Path) -> str:
    """Unique pose name: Boltz reuses filenames across seeds (see analyze_pool)."""
    seed = next((p for p in src.parts if p.startswith("full_seed")), "seedX")
    return f"{seed}_{src.stem}"


def write_submission_pdb(src: Path, dst: Path) -> None:
    """Rename the heme chain's residue LIG -> HEM; keep the ligand as LIG."""
    st = gemmi.read_structure(str(src))
    st.setup_entities()
    for ch in st[0]:
        if ch.name == "B":
            for res in ch:
                res.name = "HEM"
    st.write_pdb(str(dst))


def failed_checks(pdb: Path, smiles: str) -> tuple[list[str], str]:
    """The organizers' PoseBusters path, verbatim in structure."""
    pred_mol = Chem.MolFromPDBFile(str(pdb), removeHs=False, sanitize=False)
    if pred_mol is None:
        return ["PARSE_FAIL"], "rdkit could not parse"

    fragments = Chem.SplitMolByPDBResidues(pred_mol)
    ligand = fragments.pop("LIG", None)
    pb_protein = None
    for frag in fragments.values():
        pb_protein = frag if pb_protein is None else Chem.CombineMols(pb_protein, frag)
    if ligand is None or pb_protein is None:
        return ["NO_LIG"], "no LIG residue / no protein"

    ligand = dedupe_ligand_copies(ligand)
    template = Chem.MolFromSmiles(smiles)
    note = ""
    try:
        ligand = assign_bond_orders_from_template(template, ligand)
    except ValueError as exc:
        note = f"bond order fail: {exc}"

    res = PoseBusters(config="dock").bust(mol_pred=ligand, mol_true=template, mol_cond=pb_protein)
    bools = [c for c in res.columns if res[c].dtype == bool]
    row = res[bools].iloc[0]
    return sorted(row.index[~row].tolist()), note


def main(out_dir: str, *pred_dirs: str) -> None:
    out = Path(out_dir)
    sub_dir = out / "submission_pdbs"
    sub_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pred_dir in pred_dirs:
        for src in sorted(Path(pred_dir).rglob("*_model_*.pdb")):
            mid = src.name.split("_model_")[0]
            if mid not in SMILES.index:
                continue
            pid = pose_id(src)
            dst = sub_dir / f"{pid}.pdb"
            write_submission_pdb(src, dst)
            failed, note = failed_checks(dst, SMILES[mid])
            rows.append(dict(Molecule_Name=mid, pose=pid,
                             pb_official_pass=not failed,
                             n_failed=len(failed),
                             failed=",".join(failed), note=note,
                             sub_pdb=str(dst)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "pb_official.csv", index=False)

    print(f"{df.pb_official_pass.sum()}/{len(df)} poses pass the official check "
          f"({df.pb_official_pass.mean():.0%})\n")
    print(df.groupby("Molecule_Name").pb_official_pass.agg(["sum", "count"]).to_string())
    bad = df.loc[~df.pb_official_pass, "failed"].value_counts()
    if len(bad):
        print("\nfailure modes:"); print(bad.to_string())
    notes = df.loc[df.note != "", ["pose", "note"]]
    if len(notes):
        print("\nnotes:"); print(notes.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
