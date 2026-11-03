"""Inspect predicted CYP3A4 complexes: ligand-Fe geometry + PoseBusters 'dock' checks.

Usage: python inspect_predictions.py <dir of .cif/.pdb> <out_csv> [ligand_resname]
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


def split_complex(path, lig_resname):
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_hydrogens()
    lig = gemmi.Structure(); lig.add_model(gemmi.Model("1"))
    prot = gemmi.Structure(); prot.add_model(gemmi.Model("1"))
    fe = None
    for ch in st[0]:
        lc, pc = gemmi.Chain(ch.name), gemmi.Chain(ch.name)
        for res in ch:
            if res.name == lig_resname:
                lc.add_residue(res)
            else:
                pc.add_residue(res)
                if res.name == "HEM":
                    fe = np.array(res["FE"][0].pos.tolist())
        if len(lc): lig[0].add_chain(lc)
        if len(pc): prot[0].add_chain(pc)
    return lig, prot, fe


def main(pred_dir, out_csv, lig_resname="LIG1"):
    pred_dir, out_dir = Path(pred_dir), Path(out_csv).parent / "split"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, pb_inputs = [], []
    for f in sorted(list(pred_dir.glob("*.cif")) + list(pred_dir.glob("*.pdb"))):
        mid = next(m for m in SMILES.index if m.lower() in f.name.lower())
        lig, prot, fe = split_complex(f, lig_resname)
        lig_mol = Chem.MolFromPDBBlock(lig.make_pdb_string(), removeHs=True, sanitize=False)
        lig_mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(SMILES[mid]), lig_mol)
        lig_sdf, prot_pdb = out_dir / f"{mid}_lig.sdf", out_dir / f"{mid}_prot.pdb"
        Chem.MolToMolFile(lig_mol, str(lig_sdf))
        prot.write_pdb(str(prot_pdb))
        xyz = lig_mol.GetConformer().GetPositions()
        d = np.linalg.norm(xyz - fe, axis=1)
        i = int(d.argmin()); a = lig_mol.GetAtomWithIdx(i)
        rows.append(dict(Molecule_Name=mid, min_fe_dist=d.min(), closest_atom=a.GetSymbol(),
                         closest_aromatic=a.GetIsAromatic(), n_heavy=lig_mol.GetNumHeavyAtoms()))
        pb_inputs.append((mid, lig_sdf, prot_pdb))
    df = pd.DataFrame(rows).set_index("Molecule_Name")
    pb = PoseBusters(config="dock")
    res = pd.concat([pb.bust([str(l)], None, str(p)).assign(Molecule_Name=m) for m, l, p in pb_inputs]).set_index("Molecule_Name")
    checks = res.select_dtypes(bool)
    df["pb_pass"] = checks.all(axis=1)
    df["pb_failed"] = checks.apply(lambda r: ",".join(c for c, v in r.items() if not v), axis=1)
    df.to_csv(out_csv)
    print(df.round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
