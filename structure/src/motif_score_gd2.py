"""Score the GalaxyDock2-HEME pose pool against the CYP3A4 crystal motif.

Lets Boltz-2, GD2-HEME docking and the 96 known crystal structures be compared
on one footing. The motif score is frame-independent (it uses contacts and the
Fe distance computed inside each structure), so GD2's poses can be scored
directly even though they live in the seed-1 medoid receptor frames rather than
the frames of the 3600-pose Boltz pool.

GD2 run dirs hold `receptor.pdb` (protein chain A + heme as HEM 601) and
`GD2_HEME_fb.mol2` (50 docked ligand poses), so the ligand coordinates are
combined with that receptor's protein atoms here rather than re-read from a
complex file.

Usage:
    python motif_score_gd2.py OUT_CSV [--rundir ~/gd2r]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
_M = pickle.load(open(ROOT / "structure/params/cyp3a4_crystal_motif.pkl", "rb"))
CONSENSUS = {r[2] for r in _M["consensus"]}
FREQ = {r[2]: n / _M["n"] for r, n in _M["counts"].items()}
CONTACT_CUTOFF = 4.5
HEME_RESNUM = 601
FE_LO, FE_HI = 2.29, 2.60
RDLogger.DisableLog("rdApp.*")


def read_receptor(pdb: Path):
    prot, fe = [], None
    for ln in pdb.read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name = ln[12:16].strip()
        resnum = int(ln[22:26])
        xyz = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        if resnum == HEME_RESNUM:
            if name == "FE":
                fe = xyz
        elif not name.startswith("H"):
            prot.append((resnum, xyz))
    return prot, fe


def read_poses(mol2: Path) -> list[np.ndarray]:
    out = []
    for b in mol2.read_text().split("@<TRIPOS>MOLECULE"):
        if not b.strip():
            continue
        m = Chem.MolFromMol2Block("@<TRIPOS>MOLECULE" + b, sanitize=False, removeHs=False)
        if m is None:
            continue
        xyz = m.GetConformer().GetPositions()
        heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
        out.append(xyz[heavy])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv")
    ap.add_argument("--rundir", default=str(Path.home() / "gd2r"))
    args = ap.parse_args()
    run = Path(args.rundir).expanduser()

    wsum = sum(FREQ.get(r, 0.0) for r in CONSENSUS)
    rows = []
    for d in sorted(run.iterdir()):
        rec, mol2 = d / "receptor.pdb", d / "GD2_HEME_fb.mol2"
        if not (rec.exists() and mol2.exists()):
            continue
        prot, fe = read_receptor(rec)
        if fe is None or not prot:
            continue
        ids = np.array([r for r, _ in prot])
        pxyz = np.stack([p for _, p in prot])
        for i, lig in enumerate(read_poses(mol2), 1):
            dmin = np.linalg.norm(pxyz[:, None] - lig[None], axis=2).min(axis=1)
            contacts = set(ids[dmin < CONTACT_CUTOFF].tolist())
            hit = contacts & CONSENSUS
            dfe = float(np.linalg.norm(lig - fe, axis=1).min())
            rows.append(dict(Molecule_Name=d.name, gd2_rank=i,
                             n_contacts=len(contacts),
                             motif_recall=len(hit) / len(CONSENSUS),
                             motif_weighted=sum(FREQ.get(r, 0.0) for r in hit) / wsum,
                             min_fe_dist=dfe,
                             fe_plausible=bool(FE_LO <= dfe <= FE_HI)))
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"{len(df)} GD2 poses over {df.Molecule_Name.nunique()} ligands\n")
    print("per-ligand motif_weighted (best / median):")
    print(df.groupby("Molecule_Name").motif_weighted.agg(["max", "median"]).round(3).to_string())
    print(f"\nGD2 overall: motif_weighted best {df.groupby('Molecule_Name').motif_weighted.max().mean():.3f} "
          f"| median {df.motif_weighted.median():.3f}")
    print(f"GD2 contacts: median {df.n_contacts.median():.0f} (crystal median 16)")
    print(f"GD2 inside crystal Fe IQR: {int(df.fe_plausible.sum())}/{len(df)} ({df.fe_plausible.mean():.0%})")
    print(f"GD2 Fe distance median: {df.min_fe_dist.median():.2f} A (crystal 2.44)")


if __name__ == "__main__":
    main()
