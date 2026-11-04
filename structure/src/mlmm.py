"""ML/MM: OrbMol for the heme site, Amber/OpenMM for protein, water and ions.

Energy (additive scheme, mechanical embedding):

    E = E_MM'(all atoms) + E_ML(region + link H)

E_MM' is the full Amber system (PME) with every MM term that lies entirely
inside the ML region removed, the same construction as openmm-ml's
``createMixedSystem``:
  * bonds / angles / torsions whose atoms are all in the region -> zeroed;
  * nonbonded pairs inside the region -> exceptions with zero charge and
    epsilon. OpenMM applies the reciprocal-space exclusion correction for
    exceptions, so PME does not leak region-region Coulomb back in.
Region <-> environment interactions keep MM Lennard-Jones and MM charges.
Terms crossing the boundary (e.g. N-CA-CB angle) stay MM.

ML region: ligand + heme + the Fe-bound CYM side chain (CB, HB2, HB3, SG).
The CA-CB bond is cut and capped by a link hydrogen at a FIXED DISTANCE

    r_L = r_CB + d * u,   u = (r_CA - r_CB) / |r_CA - r_CB|,   d = 1.09 A

so the ML energy depends on the cut bond's DIRECTION but not its LENGTH. That
is what makes the bookkeeping clean: the MM CA-CB stretch is then the only
description of that coordinate. (With fixed-RATIO placement the ML C-H stretch
also responds to |CA-CB|, so MM and ML both describe it and the bond comes out
roughly 1.6x too stiff.) Forces redistribute through the exact Jacobian

    M = (d/|dr|) (I - u u^T),   F_CA += M F_L,   F_CB += (I - M) F_L

A force on the link H along u gives M F_L = 0, so a purely radial force goes
entirely to CB and none to CA -- the expected consequence of freezing |CB-H_L|.

Which MM terms are removed: the capped ML fragment already describes every
interaction among ``region + {CA}``, so MM angles and torsions whose atoms all
lie in that set are deleted. Measured on CYM442 that is exactly the 3 angles
SG/HB2/HB3-CB-CA. Terms reaching further into the backbone (N-CA-CB, C-CA-CB,
HA-CA-CB) have no ML surrogate and stay MM, as does every torsion about CB-CA:
the link H lies ON that axis, so it cannot mimic a rotation about it.

Electrostatics: region <-> environment Coulomb is handled by PME with MM point
charges on both sides, so MM/ML electrostatics are long-range, not cut off.
What is missing is polarisation -- orbmol_v2 takes no external point-charge
input, so the protein field does not reshape the ML region's density, and the
region's MM charges (for the heme, an Fe(II) set) are not consistent with the
ML description. ``latent_charges`` can replace them; see ``ml_charges()``.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import openmm as mm
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from openmm import app, unit

sys.path.insert(0, str(Path(__file__).parent))
from orb_calculator import OrbCalculator  # noqa: E402

KJMOL_TO_EV = 1.0 / 96.485332
D_LINK = 1.09          # A, C-H bond length for the link hydrogen


def atom_symbols(topology):
    """Element symbols, falling back to the atom name.

    tleap writes an OLD-STYLE prmtop with no ``%FLAG ATOMIC_NUMBER``, so OpenMM
    cannot infer the element for metals and leaves ``atom.element is None`` on
    the heme Fe. The same gap bites sander (see qmmm_minimize.py). The atom name
    carries it, so use that.
    """
    syms = []
    for a in topology.atoms():
        if a.element is not None:
            syms.append(a.element.symbol)
            continue
        nm = "".join(c for c in a.name if c.isalpha()).capitalize()
        if nm not in ("Fe",):
            raise ValueError(f"cannot resolve element for atom {a.index} {a.name} "
                             f"in {a.residue.name}")
        syms.append(nm)
    return syms


def ml_region(topology, ligand=("LIG", "MOL")):
    """Indices of the ML region and the (CB, CA) link pairs.

    The ligand residue is named by the GAFF2 mol2's unit, which antechamber
    writes as MOL; the QM/MM path renamed it to LIG. Accept both.
    """
    region, links = [], []
    if isinstance(ligand, str):
        ligand = (ligand,)
    for res in topology.residues():
        if res.name in tuple(ligand) + ("HEM",):
            region += [a.index for a in res.atoms()]
        elif res.name == "CYM":
            at = {a.name: a.index for a in res.atoms()}
            region += [at[n] for n in ("CB", "HB2", "HB3", "SG")]
            links.append((at["CB"], at["CA"]))
    return sorted(region), links


def strip_region(system, region, links=()):
    """Remove the MM terms the capped ML fragment already describes (in place).

    ``covered`` = region + the link partners (CA). Angles and torsions whose
    atoms all lie in ``covered`` are mimicked by the link hydrogen and are
    deleted; anything reaching further into the protein is kept. The cut CA-CB
    bond is deliberately KEPT: with a fixed-distance link H the ML energy does
    not depend on |CA-CB|, so MM is its only description.

    Returns a report of what was removed, so the caller can assert on it.
    """
    R = set(region)
    covered = R | {ca for _, ca in links}
    link_pairs = {frozenset(p) for p in links}
    rep = collections.Counter()
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            for k in range(f.getNumBonds()):
                i, j, r0, _ = f.getBondParameters(k)
                if frozenset((i, j)) in link_pairs:
                    rep["bond_kept_link"] += 1
                elif i in R and j in R:
                    f.setBondParameters(k, i, j, r0, 0.0)
                    rep["bond_zeroed"] += 1
        elif isinstance(f, mm.HarmonicAngleForce):
            for k in range(f.getNumAngles()):
                i, j, l, a0, kk = f.getAngleParameters(k)
                if {i, j, l} <= covered:
                    if kk._value != 0:
                        rep["angle_zeroed_intra" if {i, j, l} <= R
                            else "angle_zeroed_linkmimic"] += 1
                    f.setAngleParameters(k, i, j, l, a0, 0.0)
        elif isinstance(f, mm.PeriodicTorsionForce):
            for k in range(f.getNumTorsions()):
                i, j, l, m, n, ph, kk = f.getTorsionParameters(k)
                if {i, j, l, m} <= covered:
                    if kk._value != 0:
                        rep["torsion_zeroed"] += 1
                    f.setTorsionParameters(k, i, j, l, m, n, ph, 0.0)
        elif isinstance(f, mm.NonbondedForce):
            for a, i in enumerate(region):
                for j in region[a + 1:]:
                    f.addException(i, j, 0.0, 1.0, 0.0, replace=True)
                    rep["nb_exception"] += 1
        elif isinstance(f, mm.CMAPTorsionForce):
            pass  # backbone only; never inside this region
    return dict(rep)


def link_positions(x, links):
    """Link-H positions at a fixed distance along CB -> CA."""
    out = []
    for cb, ca in links:
        dr = x[ca] - x[cb]
        out.append(x[cb] + D_LINK * dr / np.linalg.norm(dr))
    return np.array(out).reshape(-1, 3)


def link_force_transfer(x, links, f_link, f_out):
    """Add the link-H forces onto CA and CB through the exact Jacobian."""
    for k, (cb, ca) in enumerate(links):
        dr = x[ca] - x[cb]
        r = np.linalg.norm(dr)
        u = dr / r
        M = (D_LINK / r) * (np.eye(3) - np.outer(u, u))
        fl = f_link[k]
        f_out[ca] += M @ fl
        f_out[cb] += fl - M @ fl
    return f_out


class MLMMCalculator(Calculator):
    """ASE calculator: OpenMM (stripped MM) + OrbMol (region + link H)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, prmtop, ml_charge, ml_spin=2, model="orbmol_v2",
                 platform="CUDA", device="auto", cutoff_nm=1.0,
                 switch_nm=0.9, **kw):
        super().__init__(**kw)
        prm = app.AmberPrmtopFile(str(prmtop))
        self.topology = prm.topology
        # A switching function is not cosmetic here: with a hard LJ cutoff the
        # energy jumps as pairs cross it, so the analytic force is not exactly
        # the gradient. Measured on the buried Cys CB, that alone produced a
        # 0.018 eV/A finite-difference discrepancy (vs 0.002 elsewhere), and it
        # is what breaks energy conservation in MD.
        system = prm.createSystem(nonbondedMethod=app.PME,
                                  nonbondedCutoff=cutoff_nm * unit.nanometer,
                                  switchDistance=switch_nm * unit.nanometer,
                                  constraints=None, rigidWater=False)
        self.region, self.links = ml_region(self.topology)
        self.strip_report = strip_region(system, self.region, self.links)
        self.system = system
        integ = mm.VerletIntegrator(1.0 * unit.femtosecond)
        plat = mm.Platform.getPlatformByName(platform)
        props = {"Precision": "mixed"} if platform == "CUDA" else {}
        self.context = mm.Context(system, integ, plat, props)
        self.ml = OrbCalculator(model, device=device, charge=ml_charge, spin=ml_spin)
        self.ml_charge, self.ml_spin = ml_charge, ml_spin
        self.symbols = atom_symbols(self.topology)
        self.ml_symbols = [self.symbols[i] for i in self.region] + ["H"] * len(self.links)
        self.parts = {}

    def make_atoms(self, positions_A):
        """Full-system ase.Atoms (positions in A, periodic box from prmtop).

        Not named ``atoms``: ASE's Calculator base class keeps the last system
        in ``self.atoms``, which would shadow a method of that name.
        """
        box = self.system.getDefaultPeriodicBoxVectors()
        cell = np.array([[v[k].value_in_unit(unit.angstrom) for k in range(3)] for v in box])
        return Atoms(symbols=self.symbols, positions=positions_A, cell=cell, pbc=True)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = self.atoms.positions                                   # A
        # --- MM
        self.context.setPositions(x * 0.1)
        st = self.context.getState(getEnergy=True, getForces=True)
        e_mm = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * KJMOL_TO_EV
        f = st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer) * KJMOL_TO_EV * 0.1
        # --- ML region + link H
        xl = link_positions(x, self.links)
        reg = Atoms(self.ml_symbols, positions=np.vstack([x[self.region], xl]))
        reg.info.update(charge=self.ml_charge, spin=self.ml_spin)
        self.ml.reset()
        reg.calc = self.ml
        e_ml = reg.get_potential_energy()
        fml = reg.get_forces()
        f = f.copy()
        n = len(self.region)
        f[self.region] += fml[:n]
        link_force_transfer(x, self.links, fml[n:], f)
        self.parts = dict(e_mm=e_mm, e_ml=e_ml)
        self.results = dict(energy=e_mm + e_ml, free_energy=e_mm + e_ml, forces=f)


def ml_charges(model, batch, total_charge=None):
    """Per-atom latent charges from orbmol_v2, captured with a forward hook.

    ``model.predict()`` returns only energy / forces / stress / confidence, so
    the latent charges that feed the model's own ``coulomb_module`` are not
    otherwise reachable. Hooking the head that produces them gives charges
    consistent with the ML description of the region -- unlike the MM charges,
    which for the heme are an Fe(II) set.

    Returns ``None`` if this model has no such head (orb_v3_omol does not).

    Caveat for dynamics: these charges depend on geometry, so using them in the
    MM Coulomb term adds a dQ/dx contribution to the true gradient. Fine for
    single points and interaction energies; for MD or minimisation hold them
    fixed over a segment rather than updating every step.
    """
    cand = [(n, m) for n, m in model.named_modules() if n.endswith("latent_charges")]
    if not cand:
        return None
    name, mod = cand[0]
    grabbed = {}

    def hook(_m, _inp, out):
        grabbed["q"] = out

    # No torch.no_grad() here: these are conservative models whose forces come
    # from autograd, so disabling grad makes predict() raise.
    h = mod.register_forward_hook(hook)
    try:
        model.predict(batch)
    finally:
        h.remove()
    q = grabbed.get("q")
    if q is None:
        return None
    if isinstance(q, dict):
        q = next(iter(q.values()))
    q = q.detach().float().cpu().numpy().ravel()
    if total_charge is not None and len(q):
        q = q + (total_charge - q.sum()) / len(q)      # enforce the known total
    return q


def premin_mm(prmtop, inpcrd, restrain_mask_heavy=True, k=400.0, platform="CUDA"):
    """Pure-MM relaxation of water/ions/H with solute heavy atoms restrained.

    Uses the unmodified Amber system (heme MM parameters exist), so solvent
    clashes from solvateOct are removed before the expensive ML/MM steps.
    Returns positions in A.
    """
    prm = app.AmberPrmtopFile(str(prmtop)); crd = app.AmberInpcrdFile(str(inpcrd))
    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                              constraints=None, rigidWater=False)
    rest = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
    rest.addGlobalParameter("k", k)
    for p in ("x0", "y0", "z0"):
        rest.addPerParticleParameter(p)
    pos = crd.getPositions(asNumpy=True)
    solvent = {"WAT", "HOH", "Na+", "Cl-", "NA", "CL"}
    syms = atom_symbols(prm.topology)
    for a in prm.topology.atoms():
        if a.residue.name not in solvent and syms[a.index] != "H":
            rest.addParticle(a.index, pos[a.index].value_in_unit(unit.nanometer))
    system.addForce(rest)
    if crd.boxVectors is not None:
        system.setDefaultPeriodicBoxVectors(*crd.boxVectors)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform))
    ctx.setPositions(pos)
    if crd.boxVectors is not None:
        ctx.setPeriodicBoxVectors(*crd.boxVectors)
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 2000)
    return ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
