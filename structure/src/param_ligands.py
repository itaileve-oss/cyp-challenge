"""GAFF2 / AM1-BCC parameters for all 20 challenge ligands (for MM-PBSA).

Parameterisation is per LIGAND, not per pose: atom types and charges are reused
across every pose of that ligand, so this is 20 antechamber runs (~2 min each,
run in parallel) rather than one per pose. That is what makes an MM-PBSA pass
over a few thousand poses tractable at all.

Hydrogens are added with RDKit from the pose geometry (addCoords) after bond
orders are assigned from the verbatim challenge SMILES - AM1-BCC needs explicit
hydrogens, and Boltz emits heavy atoms only. The SMILES is used as given, with
no protonation change, for the same reason as everywhere else in this project.

Usage:
    python param_ligands.py SELECTED_CSV OUT_DIR [--workers N]
"""
import argparse
import subprocess as sp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
RDLogger.DisableLog("rdApp.*")


def one(job: tuple) -> dict:
    mid, sdf, smiles, out_dir = job
    d = Path(out_dir) / mid
    d.mkdir(parents=True, exist_ok=True)
    res = dict(Molecule_Name=mid)
    try:
        m = Chem.MolFromMolFile(str(sdf), sanitize=False)
        m = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), m)
        Chem.SanitizeMol(m)
        mh = Chem.AddHs(m, addCoords=True)
        charge = Chem.GetFormalCharge(mh)
        Chem.MolToMolFile(mh, str(d / "lig_h.sdf"))
    except Exception as exc:  # noqa: BLE001
        return {**res, "ok": False, "err": f"prep: {exc}"[:200]}

    try:
        sp.run(["antechamber", "-i", "lig_h.sdf", "-fi", "mdl",
                "-o", "lig.gaff2.mol2", "-fo", "mol2", "-at", "gaff2",
                "-c", "bcc", "-nc", str(charge), "-pf", "y"],
               cwd=str(d), stdout=sp.PIPE, stderr=sp.STDOUT, text=True,
               timeout=1800, check=True)
        sp.run(["parmchk2", "-i", "lig.gaff2.mol2", "-f", "mol2",
                "-o", "lig.frcmod", "-s", "gaff2"],
               cwd=str(d), stdout=sp.PIPE, stderr=sp.STDOUT, text=True,
               timeout=600, check=True)
    except Exception as exc:  # noqa: BLE001
        return {**res, "ok": False, "err": f"antechamber/parmchk2: {exc}"[:200]}

    frcmod = (d / "lig.frcmod").read_text()
    return {**res, "ok": True, "err": "", "formal_charge": charge,
            "attn": frcmod.count("ATTN"), "n_atoms": mh.GetNumAtoms()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("selected_csv")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    sel = pd.read_csv(args.selected_csv)
    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    feats = pd.read_csv(Path(args.selected_csv).parent / "pose_features.csv").set_index("pose")

    jobs = []
    for _, r in sel.iterrows():
        mid = r["Molecule_Name"]
        sdf = feats.loc[r["pose"], "lig_sdf"]
        sdf = sdf if Path(sdf).is_absolute() else ROOT / "structure" / sdf
        jobs.append((mid, sdf, smi[mid], args.out_dir))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(one, jobs):
            rows.append(res)
            print(f"[{'ok' if res.get('ok') else 'FAIL'}] {res['Molecule_Name']} "
                  f"{res.get('err','')}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.out_dir) / "ligand_params.csv", index=False)
    print(f"\n{int(df.ok.sum())}/{len(df)} parameterised")
    if "attn" in df:
        bad = df[df.attn.fillna(0) > 0]
        print(f"ligands with missing (ATTN) parameters: {len(bad)}")
        if len(bad):
            print(bad[["Molecule_Name", "attn"]].to_string(index=False))


if __name__ == "__main__":
    main()
