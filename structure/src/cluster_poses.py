"""Frame-corrected pose clustering and selection.

Boltz-2 emits every diffusion sample in its own arbitrary rigid-body frame
(measured: 34.5 A raw CA RMSD between samples of one ligand, 1.4 A after
superposition - see protein_variability.py). Comparing ligand coordinates
directly therefore measures the frame, not the pose, so every pose is first
superposed onto a reference sample by its CA atoms and the ligand is carried
along by that transform.

Ligand RMSD is then symmetry-aware (rdMolAlign.CalcRMS) and computed in place,
with NO further alignment: after the CA fit the poses share a receptor frame,
and aligning the ligands to each other would hide exactly the translation that
distinguishes one binding mode from another.

Poses are clustered by the Butina leader algorithm at a 2 A cutoff (the
conventional "same pose" threshold) and the medoid of the largest cluster is
selected: a real occupied position, unlike a global medoid, which on a
multi-modal pool can land in empty space between two modes.

Usage:
    python cluster_poses.py ANALYSIS_DIR [CUTOFF]
"""
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

ROOT = Path(__file__).resolve().parents[2]
RDLogger.DisableLog("rdApp.*")
COORDINATING_N = Chem.MolFromSmarts("[n;X2;H0]")


def ca_and_ligand(pdb: Path):
    """CA coords (by residue number) and the ligand block, chain C."""
    st = gemmi.read_structure(str(pdb))
    st.remove_hydrogens()
    ca = {}
    for ch in st[0]:
        if ch.name == "A":
            for res in ch:
                for atom in res:
                    if atom.name == "CA":
                        ca[res.seqid.num] = np.array(atom.pos.tolist())
    return ca


def fit(p: np.ndarray, q: np.ndarray):
    """Kabsch transform taking frame p onto frame q."""
    pm, qm = p.mean(0), q.mean(0)
    v, _, w = np.linalg.svd((p - pm).T @ (q - qm))
    d = np.sign(np.linalg.det(v @ w))
    rot = v @ np.diag([1, 1, d]) @ w
    return rot, pm, qm


def butina(dist: np.ndarray, cutoff: float) -> np.ndarray:
    """Leader clustering: repeatedly take the pose with most neighbours."""
    n = len(dist)
    unassigned = set(range(n))
    labels = np.full(n, -1)
    cid = 0
    while unassigned:
        idx = sorted(unassigned)
        counts = {i: int(((dist[i, idx] <= cutoff)).sum()) for i in idx}
        leader = max(counts, key=lambda i: counts[i])
        members = [i for i in idx if dist[leader, i] <= cutoff]
        for m in members:
            labels[m] = cid
            unassigned.discard(m)
        cid += 1
    return labels


def main(analysis_dir: str, cutoff: float = 2.0) -> None:
    cutoff = float(cutoff)
    d = Path(analysis_dir)
    feats = pd.read_csv(d / "pose_features.csv")
    pb = pd.read_csv(d / "pb_official.csv")
    df = feats.merge(pb[["pose", "pb_official_pass", "n_failed", "sub_pdb"]], on="pose", how="left")
    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    prev_csv = d / "selected_poses.csv"
    prev = (pd.read_csv(prev_csv).set_index("Molecule_Name")["pose"].to_dict()
            if prev_csv.exists() else {})

    rows, detail = [], []
    for mid, grp in df.groupby("Molecule_Name"):
        can_coord = bool(Chem.MolFromSmiles(smi[mid]).GetSubstructMatches(COORDINATING_N))
        pool = grp[grp["pb_official_pass"] == True]  # noqa: E712
        stage = "pb_valid"
        if pool.empty:
            pool, stage = grp.nsmallest(1, "n_failed"), "NO_VALID_POSE"
        elif can_coord:
            coord = pool[pool["fe_coordinated"]]
            if not coord.empty:
                pool, stage = coord, "pb_valid+fe_coord"
        # Poses whose bond-order assignment failed in analyze_pool have no
        # lig_sdf (NaN), yet still pass the official check, which uses the
        # organizers' more tolerant matcher. They cannot be clustered, so drop
        # them here rather than feeding NaN to RDKit.
        pool = pool[pool["lig_sdf"].notna()]
        pool = pool[pool["lig_sdf"].map(lambda p: Path(str(p)).exists())]
        pool = pool.reset_index(drop=True)
        if pool.empty:
            print(f"[skip] {mid}: no clusterable pose (missing ligand SDFs)")
            continue

        # superpose every pose onto the first, carrying the ligand along
        ref_ca = ca_and_ligand(Path(pool.loc[0, "path"]))
        mols, keep = [], []
        for idx, r in pool.iterrows():
            ca = ca_and_ligand(Path(r["path"]))
            common = sorted(set(ca) & set(ref_ca))
            rot, pm, qm = fit(np.stack([ca[i] for i in common]),
                              np.stack([ref_ca[i] for i in common]))
            m = Chem.MolFromMolFile(str(r["lig_sdf"]), sanitize=False)
            if m is None:
                continue
            conf = m.GetConformer()
            xyz = (conf.GetPositions() - pm) @ rot + qm
            for i, pos in enumerate(xyz):
                conf.SetAtomPosition(i, pos.tolist())
            mols.append(m)
            keep.append(idx)

        # mols must stay index-aligned with pool, so drop any row whose ligand
        # could not be read rather than letting the two lists drift apart.
        pool = pool.loc[keep].reset_index(drop=True)
        if pool.empty:
            print(f"[skip] {mid}: no readable ligand geometry")
            continue

        n = len(mols)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    v = rdMolAlign.CalcRMS(mols[i], mols[j])
                except Exception:
                    v = float(np.sqrt(((mols[i].GetConformer().GetPositions()
                                        - mols[j].GetConformer().GetPositions()) ** 2)
                                      .sum(axis=1).mean()))
                dist[i, j] = dist[j, i] = v

        labels = butina(dist, cutoff) if n > 1 else np.zeros(1, int)
        sizes = pd.Series(labels).value_counts()
        top = sizes.index[0]
        members = np.where(labels == top)[0]
        sub = dist[np.ix_(members, members)]
        medoid = members[sub.mean(axis=1).argmin()] if len(members) > 1 else members[0]
        pick = pool.loc[medoid]

        off = dist[np.triu_indices(n, 1)] if n > 1 else np.array([np.nan])
        rows.append(dict(
            Molecule_Name=mid, stage=stage, n_pool=n,
            mean_rmsd_aligned=round(float(np.nanmean(off)), 2),
            n_clusters=int(sizes.size), top_cluster=int(sizes.iloc[0]),
            top_frac=round(float(sizes.iloc[0] / n), 2),
            singletons=int((sizes == 1).sum()),
            pose=pick["pose"], ligand_iptm=round(pick["ligand_iptm"], 3),
            min_fe_dist=round(pick["min_fe_dist"], 2),
            changed_vs_prev=pick["pose"] != prev.get(mid),
            sub_pdb=pick["sub_pdb"],
        ))
        detail.append(pd.DataFrame(dict(Molecule_Name=mid, pose=pool["pose"],
                                        cluster=labels)))

    out = pd.DataFrame(rows)
    out.to_csv(d / "selected_poses_clustered.csv", index=False)
    pd.concat(detail).to_csv(d / "pose_clusters.csv", index=False)
    with pd.option_context("display.width", 250):
        print(out.drop(columns="sub_pdb").to_string(index=False))
    print(f"\ncutoff {cutoff} A | picks changed vs un-aligned scoring: "
          f"{int(out.changed_vs_prev.sum())}/{len(out)}")
    print(f"mean aligned ligand RMSD within pool: {out.mean_rmsd_aligned.mean():.2f} A "
          f"| mean top-cluster occupancy: {out.top_frac.mean():.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:])
