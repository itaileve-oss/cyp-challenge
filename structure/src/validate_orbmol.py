"""Validate OrbMol on a fully protonated CYP3A4 heme site.

Checks, in order of how badly a failure would mislead downstream work:
  1. geometry sanity of the protonated system (no H clashes, right H count)
  2. forces are the gradient of the energy (finite-difference check) -- required
     for MD / metadynamics
  3. Fe(III) spin-state ordering (doublet / quartet / sextet)
  4. vertical interaction energy ligand <-> heme site
  5. relax the ligand with the heme site frozen and measure Fe-N against the
     crystal range 2.20-2.35 A (sander PM6 gave 1.96 A on this pose)

Usage:
    python validate_orbmol.py POSE_PDB LIG_H_SDF [--model orbmol_v2] [--device auto]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orb_calculator import OrbCalculator, interaction_energy  # noqa: E402
from orb_systems import heme_site                             # noqa: E402

EV2KCAL = 23.060548


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose")
    ap.add_argument("sdf")
    ap.add_argument("--model", default="orbmol_v2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = {"pose": a.pose, "model": a.model}

    at = heme_site(a.pose, a.sdf)
    q = at.info["charge"]
    sym = np.array(at.get_chemical_symbols())
    d = at.get_all_distances()
    np.fill_diagonal(d, 9)
    hh = d[np.ix_(sym == "H", sym == "H")].min()
    hx = d[np.ix_(sym == "H", sym != "H")].min()
    rep["geometry"] = dict(n_atoms=len(at), n_H=int((sym == "H").sum()),
                           charge=q, min_HH=round(float(hh), 3),
                           min_H_heavy=round(float(hx), 3))
    print("geometry", rep["geometry"])

    calc = OrbCalculator(a.model, device=a.device, charge=q, spin=2)

    # 2. finite-difference force check on 6 random atoms x 3 components
    at.info["spin"] = 2
    at.calc = calc
    t0 = time.time(); e0 = at.get_potential_energy(); t_first = time.time() - t0
    calc.reset(); t0 = time.time(); f = at.get_forces(); t_call = time.time() - t0
    rng = np.random.default_rng(0)
    h, errs = 1e-3, []
    for i in rng.choice(len(at), 6, replace=False):
        for k in range(3):
            p = at.positions.copy()
            ep = []
            for s in (+1, -1):
                at.positions = p; at.positions[i, k] += s * h
                ep.append(at.get_potential_energy())
            at.positions = p
            errs.append((-(ep[0] - ep[1]) / (2 * h), f[i, k]))
    errs = np.array(errs)
    rep["fd_check"] = dict(max_abs_err_eV_A=float(np.abs(errs[:, 0] - errs[:, 1]).max()),
                           max_force=float(np.abs(f).max()),
                           ms_per_call=round(t_call * 1000), first_call_s=round(t_first, 1))
    print("fd_check", rep["fd_check"])

    # 3. spin states (complex)
    spins = {}
    for mult in (2, 4, 6):
        at.info["spin"] = mult; calc.reset()
        spins[mult] = at.get_potential_energy()
    ref = min(spins.values())
    rep["spin_kcal_rel"] = {m: round((e - ref) * EV2KCAL, 2) for m, e in spins.items()}
    print("spin states (kcal/mol rel.)", rep["spin_kcal_rel"])
    best = min(spins, key=spins.get)

    # 4. interaction energy at the pose, in the lowest spin state
    ie = interaction_energy(calc, at, charge_complex=q,
                            charge_ligand=at.info["charge_ligand"],
                            charge_host=at.info["charge_host"],
                            spin_complex=best, spin_ligand=1, spin_host=best)
    rep["dE_int_kcal"] = round(ie["dE_kcal"], 2)
    print("dE_int (kcal/mol)", rep["dE_int_kcal"])

    # 5. constrained relaxation: heme site frozen, ligand free
    from ase.constraints import FixAtoms
    from ase.optimize import FIRE

    fe = int(np.where(sym == "Fe")[0][0])
    lig = np.where(at.get_tags() == 1)[0]
    ligN = [i for i in lig if sym[i] in ("N", "O", "S")]
    coord = min(ligN, key=lambda i: at.get_distance(fe, i))
    r0 = at.get_distance(fe, coord)
    at.info["spin"] = best; calc.reset()
    at.set_constraint(FixAtoms(indices=np.where(at.get_tags() == 0)[0]))
    t0 = time.time()
    opt = FIRE(at, logfile=None)
    conv = opt.run(fmax=0.05, steps=a.steps)
    rep["relax"] = dict(coord_atom=f"{sym[coord]}{coord}", fe_x_start=round(r0, 3),
                        fe_x_end=round(at.get_distance(fe, coord), 3),
                        converged=bool(conv), steps=opt.nsteps,
                        dE_relax_kcal=round((at.get_potential_energy() - spins[best]) * EV2KCAL, 2),
                        wall_s=round(time.time() - t0, 1))
    print("relax", rep["relax"])

    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2))
        at.write(str(Path(a.out).with_suffix(".xyz")))


if __name__ == "__main__":
    main()
