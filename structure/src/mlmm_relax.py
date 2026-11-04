"""Relax a pose with ML/MM: OrbMol heme site inside Amber protein and water.

This is the environment-aware counterpart to ``validate_orbmol.py``, which runs
the same heme site in vacuum. The observable both aim at is the Fe-ligand
coordination distance, where there is an external reference: 2.20-2.35 A across
deposited CYP3A4 type-II complexes, against 1.96 A from sander/PM6.

Only the ligand moves. Protein, water and heme stay fixed, which keeps the
problem to a few dozen free coordinates at ~2 s per ML/MM energy and makes the
Fe-N result attributable to the potential rather than to a 83k-atom relaxation.

Usage:
    python mlmm_relax.py BUILD_DIR [--steps 200] [--charge -2] [--spin 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from openmm import app

sys.path.insert(0, str(Path(__file__).parent))
from mlmm import MLMMCalculator, atom_symbols, premin_mm  # noqa: E402

EV2KCAL = 23.060548
DONORS = ("N", "O", "S")


def fe_contact(at, syms, fe, ligand_idx):
    """(index, element, distance) of the ligand donor closest to Fe."""
    cand = [i for i in ligand_idx if syms[i] in DONORS]
    if not cand:
        cand = list(ligand_idx)
    i = min(cand, key=lambda j: at.get_distance(fe, j, mic=True))
    return i, syms[i], at.get_distance(fe, i, mic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--charge", type=int, default=-2)
    ap.add_argument("--spin", type=int, default=2)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--free", choices=("ligand", "region"), default="region",
                    help="relax the ligand only, or the whole ML region "
                         "(ligand + heme + Cys cap). 'ligand' leaves the iron "
                         "frozen, so Fe-N can only close from one side.")
    ap.add_argument("--premin-k", type=float, default=4000.0,
                    help="restraint on solute heavy atoms during the MM "
                         "pre-minimisation, kJ/mol/nm^2. Too soft and MM -- "
                         "which has no Fe-N bond -- pulls the donor away "
                         "before the ML potential ever sees the pose.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.build_dir)
    rep = {"build_dir": str(d), "ml_charge": a.charge, "ml_spin": a.spin}

    t0 = time.time()
    x = premin_mm(d / "solv.prmtop", d / "solv.inpcrd", k=a.premin_k,
                  platform=a.platform)
    rep["mm_premin_s"] = round(time.time() - t0, 1)

    calc = MLMMCalculator(str(d / "solv.prmtop"), ml_charge=a.charge,
                          ml_spin=a.spin, platform=a.platform)
    rep["strip_report"] = calc.strip_report
    rep["n_atoms"] = len(calc.symbols)
    rep["n_ml_region"] = len(calc.region) + len(calc.links)

    prm = app.AmberPrmtopFile(str(d / "solv.prmtop"))
    syms = atom_symbols(prm.topology)
    ligand = [at.index for at in prm.topology.atoms() if at.residue.name in ("MOL", "LIG")]
    fe = next(i for i in calc.region if syms[i] == "Fe")

    at = calc.make_atoms(x)
    at.calc = calc
    free = set(ligand) if a.free == "ligand" else set(calc.region)
    at.set_constraint(FixAtoms(indices=[i for i in range(len(at)) if i not in free]))
    rep["free_selection"] = a.free
    rep["n_free"] = len(free)
    rep["premin_k"] = a.premin_k

    e0 = at.get_potential_energy()
    i_d, el, r0 = fe_contact(at, syms, fe, ligand)
    rep["donor"] = f"{el}{i_d}"
    rep["fe_donor_start"] = round(float(r0), 3)
    rep["e_start"] = dict(total=round(e0, 4), mm=round(calc.parts["e_mm"], 4),
                          ml=round(calc.parts["e_ml"], 4))
    print(f"{rep['n_atoms']} atoms | ML region {rep['n_ml_region']} | free {len(free)}")
    print(f"start: Fe-{rep['donor']} {r0:.3f} A | E {e0:.3f} eV", flush=True)

    t0 = time.time()
    opt = FIRE(at, logfile="-")
    conv = bool(opt.run(fmax=a.fmax, steps=a.steps))
    e1 = at.get_potential_energy()
    r1 = at.get_distance(fe, i_d, mic=True)
    rep.update(converged=conv, steps=opt.nsteps, relax_s=round(time.time() - t0, 1),
               fe_donor_end=round(float(r1), 3),
               dE_relax_kcal=round((e1 - e0) * EV2KCAL, 2),
               rmsd_ligand=round(float(np.sqrt(
                   ((at.positions[ligand] - x[ligand]) ** 2).sum(1).mean())), 3))
    print(f"\nend:   Fe-{rep['donor']} {r1:.3f} A  (crystal 2.20-2.35, PM6 1.96)")
    print(f"       dE {rep['dE_relax_kcal']:+.2f} kcal/mol | ligand RMSD "
          f"{rep['rmsd_ligand']:.3f} A | {opt.nsteps} steps in {rep['relax_s']}s")

    out = Path(a.out) if a.out else d / "mlmm_relax.json"
    out.write_text(json.dumps(rep, indent=2))
    at[sorted(calc.region)].write(str(out.with_suffix(".xyz")))   # region covers the ligand
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
