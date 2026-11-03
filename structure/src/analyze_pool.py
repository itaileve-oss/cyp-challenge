"""Per-pose sanity + selection features for a Boltz-2 CYP3A4 pose pool.

For every pose: heme-Fe coordination, which face of the heme the ligand sits
on (distal = orthosteric, opposite Cys442), contacts with orthosteric pocket
residues, Boltz confidence metrics and PoseBusters validity.

Chains: A = protein, B = heme, C = ligand (both B and C are named LIG).

Usage:
    python analyze_pool.py OUT_DIR PRED_DIR [PRED_DIR ...]
"""
import json
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

# CYP3A4 distal (orthosteric) pocket residues, from ketoconazole/ritonavir/
# midazolam complexes (2V0M, 3NXU, 5TE8): F-cluster, I-helix, SRS1/4/5/6.
POCKET = {57, 105, 108, 119, 120, 212, 213, 215, 241, 301, 304, 305, 309,
          369, 370, 372, 374, 482}
CONTACT_CUTOFF = 4.5
FE_COORD_CUTOFF = 2.6
CONF_KEYS = ["confidence_score", "iptm", "ligand_iptm", "complex_plddt",
             "complex_iplddt", "complex_ipde"]


def pose_id(path: Path) -> str:
    """Globally unique pose name.

    Every Boltz pass writes the same filenames (`<MOL>_model_0..19`), so the
    bare stem collides across seeds: merging on it multiplied rows ~9x and the
    per-pose split/submission files overwrote each other. Prefixing the seed
    directory makes the id unique across the whole pool.
    """
    seed = next((p for p in path.parts if p.startswith("full_seed")), "seedX")
    return f"{seed}_{path.stem}"


def heme_normal(heme_xyz: np.ndarray) -> np.ndarray:
    centered = heme_xyz - heme_xyz.mean(axis=0)
    return np.linalg.svd(centered)[2][-1]


def pose_features(path: Path, mid: str, split_dir: Path) -> dict:
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    model = st[0]

    prot_atoms, heme_xyz, lig_lines = [], [], []
    fe = sg = None
    for ch in model:
        for res in ch:
            for atom in res:
                xyz = np.array(atom.pos.tolist())
                if ch.name == "A":
                    prot_atoms.append((res.seqid.num, xyz))
                    if res.seqid.num == 442 and atom.name == "SG":
                        sg = xyz
                elif ch.name == "B":
                    heme_xyz.append(xyz)
                    if atom.element == gemmi.Element("Fe"):
                        fe = xyz

    lig = gemmi.Structure(); lig.add_model(gemmi.Model("1")); lig[0].add_chain(model["C"])
    prot = gemmi.Structure(); prot.add_model(gemmi.Model("1"))
    prot[0].add_chain(model["A"]); prot[0].add_chain(model["B"])

    pid = pose_id(path)
    row = dict(Molecule_Name=mid, pose=pid, path=str(path))
    mol = Chem.MolFromPDBBlock(lig.make_pdb_string(), removeHs=True, sanitize=False)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(SMILES[mid]), mol)
    except ValueError:
        return {**row, "bond_order_fail": True}

    lig_sdf = split_dir / f"{pid}_lig.sdf"
    prot_pdb = split_dir / f"{pid}_prot.pdb"
    Chem.MolToMolFile(mol, str(lig_sdf))
    prot.write_pdb(str(prot_pdb))

    xyz = mol.GetConformer().GetPositions()
    d_fe = np.linalg.norm(xyz - fe, axis=1)
    i = int(d_fe.argmin())
    atom = mol.GetAtomWithIdx(i)

    n = heme_normal(np.array(heme_xyz))
    proximal = np.sign((sg - fe) @ n)
    side = (xyz.mean(axis=0) - fe) @ n * -proximal  # >0 = distal face

    res_ids = np.array([r for r, _ in prot_atoms])
    pxyz = np.stack([p for _, p in prot_atoms])
    dmin = np.linalg.norm(pxyz[:, None] - xyz[None], axis=2).min(axis=1)
    contacts = set(res_ids[dmin < CONTACT_CUTOFF].tolist())

    conf = json.loads((path.parent / f"confidence_{path.stem}.json").read_text())
    return {
        **row, "bond_order_fail": False,
        "min_fe_dist": d_fe.min(),
        "fe_atom": atom.GetSymbol(),
        "fe_atom_aromatic": atom.GetIsAromatic(),
        "fe_coordinated": bool(d_fe.min() < FE_COORD_CUTOFF),
        "centroid_fe_dist": np.linalg.norm(xyz.mean(axis=0) - fe),
        "distal_height": side,
        "distal": bool(side > 0),
        "n_pocket_contacts": len(contacts & POCKET),
        "pocket_contacts": ",".join(map(str, sorted(contacts & POCKET))),
        "n_contacts": len(contacts),
        **{k: conf[k] for k in CONF_KEYS},
        "lig_sdf": str(lig_sdf), "prot_pdb": str(prot_pdb),
    }


def main(out_dir: str, *pred_dirs: str) -> None:
    out = Path(out_dir)
    split_dir = out / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pred_dir in pred_dirs:
        for path in sorted(Path(pred_dir).rglob("*_model_*.pdb")):
            mid = path.name.split("_model_")[0]
            if mid in SMILES.index:
                rows.append(pose_features(path, mid, split_dir))
    df = pd.DataFrame(rows)

    ok = df[~df["bond_order_fail"]]
    pb = PoseBusters(config="dock")
    res = pd.concat([pb.bust([l], None, p).assign(pose=s)
                     for s, l, p in ok[["pose", "lig_sdf", "prot_pdb"]].itertuples(index=False)])
    checks = res.set_index("pose").select_dtypes(bool)
    df = df.merge(pd.DataFrame({
        "pb_pass": checks.all(axis=1),
        "pb_failed": checks.apply(lambda r: ",".join(c for c, v in r.items() if not v), axis=1),
    }), left_on="pose", right_index=True, how="left")
    df.to_csv(out / "pose_features.csv", index=False)

    summary = df.groupby("Molecule_Name").agg(
        n=("pose", "count"),
        bond_fail=("bond_order_fail", "sum"),
        pb_pass=("pb_pass", "sum"),
        distal=("distal", "sum"),
        fe_coord=("fe_coordinated", "sum"),
        fe_atoms=("fe_atom", lambda s: ",".join(f"{k}:{v}" for k, v in s.value_counts().items())),
        med_fe=("min_fe_dist", "median"),
        med_pocket=("n_pocket_contacts", "median"),
        iptm_max=("iptm", "max"),
        iptm_med=("iptm", "median"),
    ).round(2)
    summary.insert(1, "SMILES", SMILES.reindex(summary.index).values)
    summary.to_csv(out / "ligand_summary.csv")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(summary.drop(columns="SMILES").to_string())
    bad = df.loc[df["pb_pass"] == False, "pb_failed"].value_counts()
    if len(bad):
        print("\nPoseBusters failure modes:"); print(bad.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
