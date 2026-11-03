"""PoseBusters + heme-Fe geometry for Boltz-2 output poses.

Boltz writes the complex as three chains and names BOTH the cofactor and the
ligand "LIG", so poses are split by CHAIN, not by residue name:
    A = protein, B = heme (HEM), C = ligand.

Usage:
    python check_boltz_poses.py OUT_CSV PRED_DIR [PRED_DIR ...]
"""
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from posebusters import PoseBusters
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
SMILES = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]

PROTEIN_CHAINS = ("A", "B")  # heme stays with the receptor for contact checks
LIGAND_CHAIN = "C"


def split_complex(path: Path, workdir: Path):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_hydrogens()

    lig = gemmi.Structure(); lig.add_model(gemmi.Model("1"))
    prot = gemmi.Structure(); prot.add_model(gemmi.Model("1"))
    fe = None
    for ch in st[0]:
        target = lig if ch.name == LIGAND_CHAIN else prot
        new = gemmi.Chain(ch.name)
        for res in ch:
            new.add_residue(res)
            for atom in res:
                if atom.element == gemmi.Element("Fe"):
                    fe = np.array(atom.pos.tolist())
        if len(new):
            target[0].add_chain(new)
    return lig, prot, fe


def main(out_csv: str, *pred_dirs: str) -> None:
    workdir = Path(out_csv).parent / "split"
    workdir.mkdir(parents=True, exist_ok=True)

    rows, pb_inputs = [], []
    for pred_dir in pred_dirs:
        label = Path(pred_dir).name
        for path in sorted(Path(pred_dir).rglob("*_model_*.pdb")):
            mid = next((m for m in SMILES.index if m.lower() in path.name.lower()), None)
            if mid is None:
                continue
            lig, prot, fe = split_complex(path, workdir)
            mol = Chem.MolFromPDBBlock(lig.make_pdb_string(), removeHs=True, sanitize=False)
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(SMILES[mid]), mol)
            except ValueError:
                rows.append(dict(config=label, pose=path.stem, Molecule_Name=mid,
                                 bond_order_fail=True))
                continue

            stem = f"{label}_{path.stem}"
            lig_sdf, prot_pdb = workdir / f"{stem}_lig.sdf", workdir / f"{stem}_prot.pdb"
            Chem.MolToMolFile(mol, str(lig_sdf))
            prot.write_pdb(str(prot_pdb))

            xyz = mol.GetConformer().GetPositions()
            d = np.linalg.norm(xyz - fe, axis=1)
            i = int(d.argmin())
            atom = mol.GetAtomWithIdx(i)
            rows.append(dict(config=label, pose=path.stem, Molecule_Name=mid,
                             bond_order_fail=False, min_fe_dist=d.min(),
                             closest_atom=atom.GetSymbol(),
                             fe_coordinated=bool(d.min() < 2.6)))
            pb_inputs.append((stem, lig_sdf, prot_pdb))

    df = pd.DataFrame(rows)
    pb = PoseBusters(config="dock")
    res = pd.concat(
        [pb.bust([str(l)], None, str(p)).assign(stem=s) for s, l, p in pb_inputs]
    ).set_index("stem")
    checks = res.select_dtypes(bool)
    df["stem"] = df.apply(lambda r: f"{r['config']}_{r['pose']}", axis=1)
    df = df.merge(
        pd.DataFrame({"pb_pass": checks.all(axis=1),
                      "pb_failed": checks.apply(lambda r: ",".join(c for c, v in r.items() if not v), axis=1)}),
        left_on="stem", right_index=True, how="left")
    df.to_csv(out_csv, index=False)

    # Pose diversity: all samples share one coordinate frame (same receptor
    # prediction), so a direct pairwise ligand RMSD is meaningful without
    # superposition. Low spread = the sampler collapsed onto one answer.
    div = {}
    for (cfg, mid), grp in df.dropna(subset=["min_fe_dist"]).groupby(["config", "Molecule_Name"]):
        coords = []
        for stem in grp["stem"]:
            m = Chem.MolFromMolFile(str(workdir / f"{stem}_lig.sdf"), sanitize=False)
            if m is not None:
                coords.append(m.GetConformer().GetPositions())
        if len(coords) > 1:
            arr = np.stack(coords)
            n = len(arr)
            d = [np.sqrt(((arr[i] - arr[j]) ** 2).sum(axis=1).mean())
                 for i in range(n) for j in range(i + 1, n)]
            div[(cfg, mid)] = float(np.mean(d))
    df["pair_rmsd"] = df.apply(lambda r: div.get((r["config"], r["Molecule_Name"]), np.nan), axis=1)

    print(df.groupby("config").agg(
        n=("pose", "count"),
        pb_pass_rate=("pb_pass", "mean"),
        fe_coord_rate=("fe_coordinated", "mean"),
        median_fe_dist=("min_fe_dist", "median"),
        bond_order_fails=("bond_order_fail", "sum"),
        mean_pair_rmsd=("pair_rmsd", "mean"),
    ).round(3).to_string())
    bad = df.loc[df["pb_pass"] == False, "pb_failed"].value_counts()
    if len(bad):
        print("\nfailure modes:"); print(bad.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
