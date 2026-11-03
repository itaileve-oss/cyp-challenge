"""Knowledge-based pose scoring against the CYP3A4 crystal binding motif.

Built from 96 active-site ligand poses in known CYP3A4 complexes
(`data/structures/cyp3a4_query_pose_analysis.pkl`, summarised into
`structure/params/cyp3a4_crystal_motif.pkl`). Unlike xTB dE or Boltz cluster
population, this criterion is grounded in experiment, so it can arbitrate
between selections that the other two disagree about.

What the crystals say (96 active-site poses):
  * Fe-ligand distance: median 2.44 A, IQR 2.29-2.60, max 6.27
  * 75% coordinate the iron (<2.6 A), and the closest atom is N in 88%
  * 16 contacted residues at the median
  * 14 residues are contacted by >=50% of poses (the consensus motif), led by
    Ser119 98%, Thr309 96%, Arg212 96%, Ile369 96%, Ala305 95%

Three scores per pose, deliberately kept separate rather than blended, so the
contribution of each is visible:
  motif_recall     fraction of the consensus motif this pose contacts
  motif_weighted   contact frequency-weighted recall (rewards hitting the
                   residues that nearly every crystal pose touches)
  fe_plausible     whether the Fe distance falls in the crystal IQR

Usage:
    python motif_score.py POSE_CSV OUT_CSV [--workers N] [--valid-only]
"""
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MOTIF_PKL = ROOT / "structure/params/cyp3a4_crystal_motif.pkl"
CONTACT_CUTOFF = 4.5
FE_LO, FE_HI = 2.29, 2.60          # crystal interquartile range
_M = pickle.load(open(MOTIF_PKL, "rb"))
CONSENSUS = {r[2] for r in _M["consensus"]}                     # residue numbers
FREQ = {r[2]: n / _M["n"] for r, n in _M["counts"].items()}     # contact frequency


def score_one(job: tuple) -> dict:
    pose, mid, pdb = job
    out = dict(pose=pose, Molecule_Name=mid)
    try:
        st = gemmi.read_structure(str(pdb))
        st.remove_hydrogens()
        model = st[0]
        lig, prot, fe = [], [], None
        for ch in model:
            for res in ch:
                for atom in res:
                    xyz = np.array(atom.pos.tolist())
                    nm = res.name
                    if nm == "LIG":
                        lig.append(xyz)
                    elif nm == "HEM":
                        if atom.element == gemmi.Element("Fe"):
                            fe = xyz
                    else:
                        prot.append((res.seqid.num, xyz))
        if not lig or fe is None or not prot:
            return {**out, "ok": False}
        lig = np.stack(lig)
        ids = np.array([r for r, _ in prot])
        pxyz = np.stack([p for _, p in prot])
        dmin = np.linalg.norm(pxyz[:, None] - lig[None], axis=2).min(axis=1)
        contacts = set(ids[dmin < CONTACT_CUTOFF].tolist())

        d_fe = float(np.linalg.norm(lig - fe, axis=1).min())
        hit = contacts & CONSENSUS
        wsum = sum(FREQ.get(r, 0.0) for r in CONSENSUS)
        whit = sum(FREQ.get(r, 0.0) for r in hit)
        return {**out, "ok": True,
                "n_contacts": len(contacts),
                "motif_recall": len(hit) / len(CONSENSUS),
                "motif_weighted": (whit / wsum) if wsum else np.nan,
                "motif_missing": ",".join(map(str, sorted(CONSENSUS - contacts))),
                "min_fe_dist": d_fe,
                "fe_plausible": bool(FE_LO <= d_fe <= FE_HI),
                "fe_dev_from_median": d_fe - 2.44}
    except Exception:  # noqa: BLE001
        return {**out, "ok": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_csv")
    ap.add_argument("out_csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--valid-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.pose_csv)
    if args.valid_only and "pb_official_pass" in df.columns:
        df = df[df["pb_official_pass"] == True]  # noqa: E712
    col = "sub_pdb" if "sub_pdb" in df.columns else "path"
    jobs = [(r["pose"], r["Molecule_Name"], r[col]) for _, r in df.iterrows()
            if isinstance(r[col], str)]
    print(f"scoring {len(jobs)} poses against the crystal motif on {args.workers} workers")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(score_one, jobs, chunksize=20):
            rows.append(res)
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    ok = out[out["ok"] == True]  # noqa: E712
    print(f"{len(ok)}/{len(out)} scored")
    if len(ok):
        print("\nper-ligand motif_weighted (best / median):")
        g = ok.groupby("Molecule_Name")["motif_weighted"].agg(["max", "median", "count"]).round(3)
        print(g.to_string())
        print(f"\nposes inside the crystal Fe IQR: {int(ok.fe_plausible.sum())}/{len(ok)} "
              f"({ok.fe_plausible.mean():.0%})")
        print(f"median motif_recall: {ok.motif_recall.median():.2f} "
              f"| median contacts: {ok.n_contacts.median():.0f} (crystal median 16)")


if __name__ == "__main__":
    main()
