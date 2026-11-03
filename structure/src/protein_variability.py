"""How much does the Boltz-predicted receptor move between samples and ligands?

Two separate questions, deliberately measured apart:

1. FRAME: raw CA RMSD with NO superposition. The pose-selection scoring
   (consensus / clustering) compares ligand coordinates directly, which is only
   valid if every sample is emitted in a common frame. If this is ~0 the frame
   is shared; if it is large, direct ligand RMSD is measuring protein motion.
2. CONFORMATION: CA RMSD AFTER Kabsch superposition, i.e. real backbone
   differences, plus distal-pocket side-chain RMSD (induced fit) measured after
   superposing on the backbone.

Usage:
    python protein_variability.py PRED_DIR [--out CSV]
"""
import itertools
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POCKET = {57, 105, 108, 119, 120, 212, 213, 215, 241, 301, 304, 305, 309,
          369, 370, 372, 374, 482}


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    pc, qc = p - p.mean(0), q - q.mean(0)
    v, _, w = np.linalg.svd(pc.T @ qc)
    d = np.sign(np.linalg.det(v @ w))
    rot = v @ np.diag([1, 1, d]) @ w
    return float(np.sqrt(((pc @ rot - qc) ** 2).sum(axis=1).mean()))


def kabsch_transform(p: np.ndarray, q: np.ndarray):
    """Rotation+translation taking frame of p onto q (fitted on CA)."""
    pm, qm = p.mean(0), q.mean(0)
    v, _, w = np.linalg.svd((p - pm).T @ (q - qm))
    d = np.sign(np.linalg.det(v @ w))
    rot = v @ np.diag([1, 1, d]) @ w
    return rot, pm, qm


def raw_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(((p - q) ** 2).sum(axis=1).mean()))


def load(path: Path):
    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    ca, pocket, heme, fe = {}, {}, [], None
    for ch in st[0]:
        for res in ch:
            for atom in res:
                xyz = np.array(atom.pos.tolist())
                if ch.name == "A":
                    if atom.name == "CA":
                        ca[res.seqid.num] = xyz
                    if res.seqid.num in POCKET and atom.element.name != "C" or (
                            res.seqid.num in POCKET and atom.name not in ("N", "CA", "C", "O")):
                        pocket[(res.seqid.num, atom.name)] = xyz
                elif ch.name == "B":
                    heme.append(xyz)
                    if atom.element == gemmi.Element("Fe"):
                        fe = xyz
    return ca, pocket, np.array(heme), fe


def main(pred_dir: str, out_csv: str = "protein_variability.csv") -> None:
    poses = {}
    for path in sorted(Path(pred_dir).rglob("*_model_*.pdb")):
        mid = path.name.split("_model_")[0]
        poses.setdefault(mid, []).append(path)

    data = {}
    for mid, paths in poses.items():
        for p in paths:
            data[(mid, p.stem)] = load(p)

    keys = sorted(data)
    common_ca = sorted(set.intersection(*[set(data[k][0]) for k in keys]))
    common_pk = sorted(set.intersection(*[set(data[k][1]) for k in keys]))
    print(f"{len(keys)} poses | {len(common_ca)} common CA | {len(common_pk)} common pocket side-chain atoms")

    ca = {k: np.stack([data[k][0][r] for r in common_ca]) for k in keys}
    pk = {k: np.stack([data[k][1][a] for a in common_pk]) for k in keys}
    fe = {k: data[k][3] for k in keys}

    rows = []
    # within ligand (sample-to-sample)
    for mid in poses:
        ks = [k for k in keys if k[0] == mid]
        for a, b in itertools.combinations(ks, 2):
            rot, pm, qm = kabsch_transform(ca[a], ca[b])
            pk_a = (pk[a] - pm) @ rot + qm
            rows.append(dict(scope="within_ligand", mid=mid,
                             raw_ca=raw_rmsd(ca[a], ca[b]),
                             fit_ca=kabsch_rmsd(ca[a], ca[b]),
                             pocket_sc=raw_rmsd(pk_a, pk[b]),
                             fe_shift=raw_rmsd(fe[a][None], fe[b][None])))
    # between ligands (model_0 of each)
    firsts = [k for k in keys if k[1].endswith("_model_0")]
    for a, b in itertools.combinations(firsts, 2):
        rot, pm, qm = kabsch_transform(ca[a], ca[b])
        pk_a = (pk[a] - pm) @ rot + qm
        rows.append(dict(scope="between_ligand", mid=f"{a[0]}|{b[0]}",
                         raw_ca=raw_rmsd(ca[a], ca[b]),
                         fit_ca=kabsch_rmsd(ca[a], ca[b]),
                         pocket_sc=raw_rmsd(pk_a, pk[b]),
                         fe_shift=raw_rmsd(fe[a][None], fe[b][None])))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print()
    print(df.groupby("scope")[["raw_ca", "fit_ca", "pocket_sc", "fe_shift"]]
          .describe().T.round(2).to_string())
    print("\nper-ligand sample-to-sample (mean):")
    w = df[df.scope == "within_ligand"].groupby("mid")[["raw_ca", "fit_ca", "pocket_sc", "fe_shift"]].mean()
    print(w.round(2).to_string())


if __name__ == "__main__":
    main(*[a for a in sys.argv[1:] if not a.startswith("--")])
