"""Build a solvated, fully protonated Amber system for ML/MM from a heavy-atom pose.

Reuses the QM/MM preparation (`qmmm_minimize.split_pose`, `retemplate_ligand`):
protein through pdb4amber --reduce + ff14SB, heme from the hemefe2
pentacoordinated prep (which supplies its hydrogens), ligand GAFF2 retemplated
onto the pose. Additions for ML/MM:

* the Fe-bound cysteine becomes CYM (thiolate, no HG). It is found by SG-Fe
  distance, because pdb4amber renumbers residues and 442 no longer means Cys442;
* truncated-octahedron TIP3P box (10 A buffer) and neutralising Na+/Cl-.

The heme's MM charges are the Fe(II) set. In ML/MM they only enter the
heme <-> environment Coulomb term (heme internals are on the MLIP), so the
missing +1 of Fe(III) is a small, known approximation of mechanical embedding.

AmberTools (pdb4amber, tleap) come from the `openfe` conda env.

Usage:
    python mlmm_build.py POSE_PDB OUT_DIR LIGPARAM_DIR [--mol-name OCNT-...]
"""
import argparse
import os
import shutil
import subprocess as sp
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qmmm_minimize import HEME_PREP, HEME_PRM, ROOT, retemplate_ligand, split_pose  # noqa: E402

AMBER_ENV = Path.home() / "miniconda3/envs/openfe"


def amber(cmd, cwd, log):
    env = dict(os.environ, AMBERHOME=str(AMBER_ENV),
               PATH=f"{AMBER_ENV / 'bin'}:{os.environ['PATH']}")
    r = sp.run(cmd, cwd=str(cwd), env=env, stdout=sp.PIPE, stderr=sp.STDOUT,
               text=True, timeout=3600)
    (Path(cwd) / log).write_text(r.stdout)
    return r.returncode == 0, r.stdout


def write_ferric_prep(src: Path, dst: Path, delta: float = 1.0) -> float:
    """Copy the heme prep, raising Fe's charge by ``delta`` (Fe(II) -> Fe(III)).

    The shipped prep is the Fe(II) pentacoordinated set, whose heme unit carries
    a net -2. We model Fe(III), where it should be -1. In the old sander QM/MM
    this was harmless (the QM region replaced the heme's charges), but in ML/MM
    the heme's MM charges act on the protein and water, and -- worse -- tleap
    neutralises the box against this total, so an uncorrected build gets the
    wrong number of counterions.

    Returns the new Fe charge.
    """
    out, newq = [], None
    for ln in src.read_text().splitlines():
        f = ln.split()
        if len(f) == 11 and f[1] == "FE" and f[2] == "FE":
            newq = float(f[10]) + delta
            ln = ln[:ln.rfind(f[10])] + f"{newq:.4f}"
        out.append(ln)
    if newq is None:
        raise ValueError(f"no FE atom line found in {src}")
    dst.write_text("\n".join(out) + "\n")
    return newq


def make_cym(prot_pdb: Path, heme_pdb: Path) -> int:
    """Rename the Fe-bound Cys to CYM and drop its HG; returns its resnum."""
    fe = next(np.array(a.pos.tolist()) for r in gemmi.read_structure(str(heme_pdb))[0][0]
              for a in r if a.name == "FE")
    st = gemmi.read_structure(str(prot_pdb))
    best = None
    for ch in st[0]:
        for r in ch:
            if r.name in ("CYS", "CYM"):
                sg = r.find_atom("SG", "*")
                if sg is None:
                    continue
                d = np.linalg.norm(np.array(sg.pos.tolist()) - fe)
                if best is None or d < best[0]:
                    best = (d, r)
    d, res = best
    if d > 3.0:
        raise ValueError(f"no Cys SG within 3 A of Fe (closest {d:.2f})")
    res.name = "CYM"
    for i in reversed(range(len(res))):
        if res[i].name == "HG":
            del res[i]
    st.write_pdb(str(prot_pdb))
    return res.seqid.num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_pdb")
    ap.add_argument("out_dir")
    ap.add_argument("ligparam_dir")
    ap.add_argument("--mol-name", default=None)
    ap.add_argument("--buffer", type=float, default=10.0)
    ap.add_argument("--ferrous", action="store_true",
                    help="keep the shipped Fe(II) heme charges instead of Fe(III)")
    a = ap.parse_args()

    d = Path(a.out_dir); d.mkdir(parents=True, exist_ok=True)
    nres, nlig = split_pose(Path(a.pose_pdb), d)
    ok, _ = amber(["pdb4amber", "-i", "prot_raw.pdb", "-o", "prot.pdb", "--reduce", "--dry"],
                  d, "pdb4amber.log")
    if not (d / "prot.pdb").exists():
        raise SystemExit("pdb4amber failed, see pdb4amber.log")
    cym = make_cym(d / "prot.pdb", d / "heme.pdb")
    print(f"protein residues {nres} | ligand heavy atoms {nlig} | CYM = residue {cym}")

    lp = Path(a.ligparam_dir)
    for f in ("lig.gaff2.mol2", "lig.frcmod"):
        shutil.copy(lp / f, d / f)
    shutil.copy(HEME_PRM, d / "HEM.prm")
    qfe = write_ferric_prep(HEME_PREP, d / "HEM.in", delta=0.0 if a.ferrous else 1.0)
    print(f"heme prep: Fe charge {qfe:+.4f} "
          f"({'Fe(II), as shipped' if a.ferrous else 'Fe(III)'})")
    smiles = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv") \
        .set_index("Molecule_Name")["SMILES"]
    mid = a.mol_name or lp.name
    retemplate_ligand(Path(a.pose_pdb), d / "lig.gaff2.mol2", smiles[mid], d / "lig_pose.mol2")

    (d / "build.leap").write_text(f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberprep HEM.in
loadamberparams HEM.prm
loadamberparams lig.frcmod
LIG = loadmol2 lig_pose.mol2
prot = loadpdb prot.pdb
heme = loadpdb heme.pdb
cplx = combine {{prot heme LIG}}
savepdb cplx dry.pdb
solvateOct cplx TIP3PBOX {a.buffer}
addIonsRand cplx Na+ 0
addIonsRand cplx Cl- 0
saveamberparm cplx solv.prmtop solv.inpcrd
savepdb cplx solv.pdb
quit
""")
    ok, out = amber(["tleap", "-f", "build.leap"], d, "tleap.log")
    if not (d / "solv.prmtop").exists():
        raise SystemExit("tleap failed:\n" + "\n".join(out.splitlines()[-25:]))
    import openmm as mm
    from openmm import app
    sysm = app.AmberPrmtopFile(str(d / "solv.prmtop")).createSystem()
    nb = next(f for f in sysm.getForces() if isinstance(f, mm.NonbondedForce))
    q = sum(nb.getParticleParameters(i)[0]._value for i in range(nb.getNumParticles()))
    qhem = sum(nb.getParticleParameters(at.index)[0]._value
               for at in app.AmberPrmtopFile(str(d / "solv.prmtop")).topology.atoms()
               if at.residue.name == "HEM")
    nion = sum(1 for r in app.AmberPrmtopFile(str(d / "solv.prmtop")).topology.residues()
               if r.name in ("Na+", "Cl-"))
    print(f"built solv.prmtop: {sysm.getNumParticles()} atoms, net MM charge {q:+.3f}, "
          f"heme net {qhem:+.3f}, {nion} counterions")


if __name__ == "__main__":
    main()
