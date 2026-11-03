"""Tier-1 pose ranking: GFN2-xTB interaction energy on a truncated active site.

Why xTB and not MM-GBSA / Egret-1 / OrbMol: half the challenge ligands bind
through an Fe-N coordination bond to the heme, and none of those methods can
treat iron (GAFF/OpenFF have no usable Fe coordination parameters, and Egret-1
and OrbMol are organic-element only). A scorer blind to Fe penalises exactly
the poses that are right - the failure already measured here with GNINA, which
found 0/20 coordinating poses for a ligand Boltz coordinates in 20/20. GFN2-xTB
is semi-empirical quantum mechanics and handles transition metals, so the
coordination bond is described rather than punished.

Model: ligand + heme + the Cys442 side chain (the axial thiolate), which
captures the coordination bond and the porphyrin contact. Wider protein
contacts are deliberately left to the PoseBusters filter and to tier 2, which
relaxes the ligand inside a larger truncated pocket.

    dE = E(ligand + heme + Cys) - E(heme + Cys) - E(ligand)

all three single points at the pose geometry (no relaxation), GFN2 with ALPB
water. The receptor differs slightly between Boltz samples, so the pocket term
is recomputed per pose rather than once per ligand.

IMPORTANT: charge and spin are fixed assumptions (Fe(III) high spin: charge -1
for Fe(3+) + porphyrin(2-) + two propionates, uhf 5). They are constant across
poses of one ligand, so WITHIN-ligand ranking is sound, but dE is NOT
comparable between different ligands.

Usage:
    python xtb_score.py POSE_CSV OUT_CSV [--workers N] [--xtb PATH] [--only MOL]
"""
import argparse
import os
import shutil
import subprocess as sp
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

HARTREE_KCAL = 627.5094740631
HEME_CHARGE = -1
HEME_UHF = 5          # Fe(III) high spin, d5
AXIAL_ATOMS = ("SG", "CB")
CYS_RESNUM = 442


def write_xyz(path: Path, atoms: list[tuple[str, np.ndarray]]) -> None:
    with path.open("w") as f:
        f.write(f"{len(atoms)}\n\n")
        for elem, xyz in atoms:
            f.write(f"{elem} {xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}\n")


def extract(pose_pdb: Path):
    st = gemmi.read_structure(str(pose_pdb))
    st.remove_hydrogens()
    model = st[0]
    lig, heme, cys = [], [], []
    for ch in model:
        for res in ch:
            for atom in res:
                item = (atom.element.name, np.array(atom.pos.tolist()))
                if ch.name == "C":
                    lig.append(item)
                elif ch.name == "B":
                    heme.append(item)
                elif ch.name == "A" and res.seqid.num == CYS_RESNUM and atom.name in AXIAL_ATOMS:
                    cys.append(item)
    return lig, heme, cys


def run_xtb(xtb: str, xyz: Path, charge: int, uhf: int, workdir: Path) -> float:
    cmd = [xtb, xyz.name, "--gfn", "2", "--chrg", str(charge), "--uhf", str(uhf),
           "--alpb", "water", "--sp"]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["XTBPATH"] = str(workdir)
    res = sp.run(cmd, cwd=str(workdir), env=env, stdout=sp.PIPE, stderr=sp.STDOUT,
                 text=True, timeout=1800)
    for ln in res.stdout.splitlines():
        if "TOTAL ENERGY" in ln:
            return float(ln.split()[3])
    raise RuntimeError(res.stdout[-800:])


def score_pose(job: tuple) -> dict:
    pose, mid, pdb, xtb = job
    out = dict(pose=pose, Molecule_Name=mid)
    tmp = Path(tempfile.mkdtemp(prefix="xtb_"))
    try:
        lig, heme, cys = extract(Path(pdb))
        if not lig or not heme:
            return {**out, "xtb_ok": False, "xtb_err": "missing ligand or heme"}
        pocket = heme + cys
        write_xyz(tmp / "complex.xyz", lig + pocket)
        write_xyz(tmp / "pocket.xyz", pocket)
        write_xyz(tmp / "lig.xyz", lig)
        e_cx = run_xtb(xtb, tmp / "complex.xyz", HEME_CHARGE, HEME_UHF, tmp)
        e_pk = run_xtb(xtb, tmp / "pocket.xyz", HEME_CHARGE, HEME_UHF, tmp)
        e_lg = run_xtb(xtb, tmp / "lig.xyz", 0, 0, tmp)
        return {**out, "xtb_ok": True, "xtb_err": "",
                "e_complex": e_cx, "e_pocket": e_pk, "e_ligand": e_lg,
                "dE_kcal": (e_cx - e_pk - e_lg) * HARTREE_KCAL}
    except Exception as exc:  # noqa: BLE001
        return {**out, "xtb_ok": False, "xtb_err": str(exc)[:300]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_csv", help="CSV with pose, Molecule_Name, sub_pdb (and optionally pb_official_pass)")
    ap.add_argument("out_csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--xtb", default=str(Path.home() / "miniconda3/envs/xtb_env/bin/xtb"))
    ap.add_argument("--only")
    ap.add_argument("--valid-only", action="store_true",
                    help="score only poses with pb_official_pass True (an invalid pose scores 0 anyway)")
    args = ap.parse_args()

    df = pd.read_csv(args.pose_csv)
    if args.valid_only and "pb_official_pass" in df.columns:
        df = df[df["pb_official_pass"] == True]  # noqa: E712
    if args.only:
        df = df[df["Molecule_Name"] == args.only]
    col = "sub_pdb" if "sub_pdb" in df.columns else "path"
    jobs = [(r["pose"], r["Molecule_Name"], r[col], args.xtb) for _, r in df.iterrows()]
    print(f"scoring {len(jobs)} poses on {args.workers} workers")

    rows, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(score_pose, jobs, chunksize=1):
            rows.append(res)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    ok = out[out["xtb_ok"]]
    print(f"\n{len(ok)}/{len(out)} scored")
    if len(ok):
        # Rank within ligand only: dE is not comparable across ligands.
        ok = ok.copy()
        ok["dE_rank"] = ok.groupby("Molecule_Name")["dE_kcal"].rank()
        print(ok.groupby("Molecule_Name")["dE_kcal"]
              .agg(["count", "min", "median", "max"]).round(1).to_string())
    bad = out[~out["xtb_ok"]]
    if len(bad):
        print("\nfailures:"); print(bad["xtb_err"].value_counts().head().to_string())


if __name__ == "__main__":
    main()
