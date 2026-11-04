# ML/MM: OrbMol heme site in Amber protein and water

The heme site on a machine-learned potential, the rest of the protein and the
solvent on a classical force field. This replaces the sander/PM6 QM/MM route
and removes its two hard limits: PM6's closed-shell-only semiempirical QM
(which forced an Fe(II) approximation) and the invented metal parameters a
fixed-charge force field needs for an Fe-N dative bond.

Code: `structure/src/mlmm_build.py` (system construction),
`structure/src/mlmm.py` (the coupled calculator),
`structure/src/mlmm_relax.py` (constrained relaxation driver).
Post-deadline tooling; not used in a submission.

## Energy expression

    E = E_MM'(all atoms, PME) + E_ML(region + link H)

`E_MM'` is the full Amber system with the MM terms the capped ML fragment
already describes removed. `E_ML` is `orbmol_v2` (see
[orbmol.md](orbmol.md)). Region to environment coupling keeps MM Lennard-Jones
and MM point charges under PME, so **MM/ML electrostatics are long-range, not
truncated**. What is absent is polarisation, not range -- see *Limits*.

ML region: ligand + heme + the Fe-bound CYM side chain (CB, HB2, HB3, SG),
121 atoms plus 1 link H for this pose.

## The covalent boundary

The CA-CB bond is cut and capped with a link hydrogen at a **fixed distance**
1.09 A along CB->CA:

    r_L = r_CB + d u,   u = (r_CA - r_CB)/|r_CA - r_CB|

so the ML energy depends on the cut bond's *direction* but not its *length*.
That is what makes the MM CA-CB stretch the only description of that
coordinate. With fixed-*ratio* placement (the obvious first choice) the ML C-H
stretch also responds to |CA-CB|, so both potentials act on it and the bond
comes out roughly 1.6x too stiff.

Forces transfer through the exact Jacobian `M = (d/|dr|)(I - u u^T)`:

    F_CA += M F_L,    F_CB += (I - M) F_L

with a built-in check: a force on the link H along `u` gives `M F_L = 0`, so a
purely radial force goes entirely to CB and none to CA, which is what freezing
|CB-H_L| must imply.

**Which MM terms are deleted.** The capped fragment describes every interaction
among `region + {CA}`, so MM angles and torsions whose atoms all lie in that
set are removed. Enumerated on CYM442, that is exactly 3 angles:
SG-CB-CA, HB2-CB-CA, HB3-CB-CA. A first implementation that removed only terms
*wholly inside* the region left all three **double counted**. Kept as MM:
N-CA-CB, C-CA-CB, HA-CA-CB (no ML surrogate exists for N, C or HA) and every
torsion about CB-CA -- those are safe by construction, because the link H lies
*on* that axis and cannot mimic a rotation about it.

Intra-region nonbonded pairs become zero-charge, zero-epsilon exceptions.
OpenMM applies the reciprocal-space exclusion correction for exceptions, so
PME does not leak region-region Coulomb back in.

A `switchDistance` of 0.9 nm is not cosmetic: with a hard LJ cutoff the energy
jumps as pairs cross it, and the analytic force is then not exactly the
gradient. On the buried Cys CB that alone produced a 0.018 eV/A
finite-difference discrepancy.

## Verification (OCNT-2395436)

83,786 atoms; 121 + 1 ML atoms. `strip_report` from the calculator:

| removed | count |
|---|---|
| intra-region bonds zeroed | 128 |
| cut bond kept as MM | 1 |
| **link-mimicked angles zeroed** | **3** |
| intra-region angles zeroed | 230 |
| torsions zeroed | 285 |
| nonbonded exceptions | 7260 = C(121,2) |

Finite-difference force check, h = 2e-3 A:

| probe | \|F\| eV/A | max err |
|---|---|---|
| CB (link atom) | 0.383 | 0.0087 |
| CA (link partner) | 0.153 | 0.0019 |
| Fe | 0.418 | 0.0017 |
| ligand | 1.273 | 0.0011 |
| nearest environment atom | 0.036 | 0.0006 |

CB's residual *falls* as h grows (0.0178 at h=1e-3 -> 0.0087 at h=2e-3, a 1/h
scaling), so it is energy noise of about 3.5e-5 eV (8e-4 kcal/mol), not a
gradient bias. Its source is not established; it is not the MLIP neighbour-list
cap, which was the first guess -- the cap is 120 and the region's busiest atom
has 68 neighbours.

Cost on a T500: MM pre-minimisation of 83k atoms ~20 s; ML/MM energy+forces
~1.8 s per call, of which ~1.3 s is OrbMol.

One caveat seen during relaxation: at one step the max force jumped from 0.059
to 1.72 eV/A while the energy still *decreased*. A force jump with no matching
energy jump is a discontinuity in the potential, not a bad step -- an edge
entering or leaving the 6 A neighbour graph, which orb rebuilds on every call.
FIRE recovered and reached a lower energy, but this is the behaviour to watch
if the ML/MM surface is ever used for dynamics rather than minimisation.

## Result: Fe-N against the crystal reference

`mlmm_relax.py` on OCNT-2395436, whose docked pose has an unambiguous
coordinating nitrogen (Fe-N16 2.460 A; next-nearest ligand atom 2.839 A):

| | Fe-N (A) | note |
|---|---|---|
| crystal reference (96 CYP3A4 complexes) | **2.20-2.35** | target |
| docked pose (GalaxyDock2-HEME) | 2.460 | input |
| after pure-MM pre-minimisation | 2.805 | MM has no Fe-N bond and its Fe(3+) LJ minimum is ~3.2 A, so it pushes the donor *out* |
| sander / PM6 QM/MM | 1.96 | collapses |
| **ML/MM, ML region free** | **2.339** | **inside the crystal range** |
| ML/MM, ligand only free | 2.691 | protocol artefact, see below |
| MLIP alone, in vacuum ([orbmol.md](orbmol.md)) | 2.19 | bottom edge |

Converged at fmax 0.045 eV/A in 206 steps (340 s), dE -44.1 kcal/mol, ligand
RMSD 0.232 A.

**The free-atom choice dominates the answer, so it is a flag, not a default.**
Relaxing only the ligand gives 2.691 A with a ligand RMSD of 0.132 A: the
ligand cannot reach the iron because the iron is frozen where the MM
pre-minimisation left it, so the bond can only close from one side. Freeing the
whole ML region (`--free region`, the default) lets both partners move and
lands inside the crystal range. For the same reason `--premin-k` defaults to
4000 kJ/mol/nm^2 rather than a soft restraint: at 400 the MM step alone dragged
Fe-N from 2.460 to 2.856 A before the ML potential ever saw the pose.

This is the one externally checkable number available, and it is the test PM6
failed. It is a single pose, so it validates the machinery rather than
establishing accuracy.

## The heme charge state, and why it matters here

The shipped heme prep (`hemefe2_pentacoordinated`) is the **Fe(II)** set: its
heme unit carries a net **-2**, and Fe itself carries **+0.016**. For the
Fe(III) state actually present in type-II complexes the unit should be -1.

Under sander/PM6 this was harmless and the QM/MM notes say so -- the QM region
replaced the heme's charges. **Under ML/MM it is not harmless**, twice over:
the heme's MM charges now act on protein and water, and tleap neutralises the
box against this total, so an uncorrected build gets the wrong counterion
count. `mlmm_build.py` therefore raises Fe by +1 before `tleap` runs
(`--ferrous` keeps the shipped values). Verified after the fix: heme net
-1.000, system net -0.002, 1 counterion.

## ML charges vs MM charges in the region

`orbmol_v2` predicts latent partial charges that feed its own Coulomb module,
but `predict()` does not return them, so `mlmm.ml_charges()` captures them with
a forward hook on the `latent_charges` head. (It must not be wrapped in
`torch.no_grad()`: these are conservative models whose forces come from
autograd.) Compared with the MM charges currently used for region-environment
Coulomb, RMS difference 0.209 e:

| atom | ML | MM | note |
|---|---|---|---|
| Fe | **+0.557** | **+0.016** | the Amber prep leaves iron essentially uncharged |
| S (Cys thiolate) | -0.444 | -0.884 | MM over-localises the negative charge |
| S (ligand) | +0.329 | **+1.459** | GAFF2 on a hypervalent sulfur |

Fragment sums (ML): heme -1.637, Cys cap -0.447, ligand +0.041 -- i.e. roughly
0.55 e of thiolate-to-iron donation, which is exactly the effect that sets the
iron's spin state and which no fixed-charge set can carry.

Substituting these is a *reparameterisation* of mechanical embedding, not a
move to electrostatic embedding. If used, renormalise to the corrected MM
region total rather than to -2, or the box stops being neutral. Because the
charges are geometry-dependent, using them in the MM Coulomb term adds a dQ/dx
term to the true gradient: fine for single points and interaction energies,
but for MD or minimisation hold them fixed over a segment.

## Limits

- **Mechanical embedding only.** `orbmol_v2` accepts no external point-charge
  field, so the protein does not polarise the ML region. This is the one thing
  sander/PM6 did that this does not. Partial compensation: the whole first
  coordination shell is *inside* the ML region.
- **No OpenMM-ML path.** `openmm-torch` needs a TorchScript module mapping
  positions to energy, graph construction included; orb builds its neighbour
  graph with NVIDIA Warp in Python, so `MLPotential.createMixedSystem()` is not
  a drop-in. Hence this ASE-side coupling.
- **tleap writes no Fe-SG bond** (the heme is loaded as a separate unit), so in
  the pure-MM description the heme is held only by nonbonded terms. Harmless
  for ML/MM, where Fe-SG lives inside the ML region, but it means the MM
  pre-minimisation needs its solute restraints.
- Only the ligand is relaxed in `mlmm_relax.py`. A full ML/MM minimisation of
  83k atoms at ~2 s per energy is not affordable on a T500.
