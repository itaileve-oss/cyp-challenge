# QM/MM minimisation with sander

Geometry refinement where the iron coordination is treated quantum-mechanically
and the protein classically. Built and working; too slow to finish within our
deadline, so no submitted pose was refined with it.

## Why QM/MM is the right tool here

A fixed-charge force field cannot hold an Fe–N dative bond, and the two obvious
parameter sets fail in opposite directions:

| model | Fe parameters | what minimisation does |
|---|---|---|
| Amber + Li/Merz Fe³⁺ 12-6 | R\* 1.386 Å, ε 0.0136 | LJ minimum at **3.21 Å** → pushes the nitrogen *away* (crystal is ~2.2–2.35 Å) |
| CHARMM36 heme | **ε = 0**, q = +0.24 | no repulsive wall, monotonic attraction → nitrogen **collapses** onto Fe |

Either way the answer is set by invented parameters rather than physics. In
QM/MM the bond length is decided by electrons, with no restraint target to
guess — and the result can then be checked against the crystal distribution as
independent validation.

It is also tractable, for the reason bulk xTB was not: the **QM region is
small** (~115 atoms) while the expensive pocket stays MM.

## Running it

```bash
python structure/src/qmmm_minimize.py POSE_PDB OUT_DIR LIGPARAM_DIR \
  --theory PM6 --maxcyc 100 --qmcharge -2 --mol-name OCNT-XXXXXXX
```

Pipeline: split the pose → `pdb4amber` → `tleap` (ff14SB + GAFF2 ligand +
pentacoordinated heme) → normalise the prmtop → `sander` QM/MM minimisation with
the protein backbone restrained at 10 kcal/mol and GB implicit solvent.

QM region is ligand + heme (`qmmask=':MOL | :HEM'`), MM is everything else.

## Four failures that each cost a debug cycle

**1. Ligand atom names.** `tleap` types the ligand by matching atom *names* to
the parameterised unit, and Boltz names atoms independently of antechamber
(`C35`/`N34`/`S44` vs `C1`/`N1`/`S1`). Loading the pose PDB directly gives
`Atom ... does not have a type → Failed to generate parameters`. Fix: keep the
GAFF2 template mol2 and transfer the pose coordinates into it, rebuilding
hydrogens with RDKit.

**2. mol2 type column.** antechamber writes **GAFF2** types (`c3`, `n7`, `s6`)
there, not SYBYL types (`C.3`, `N.ar`), so deriving the element with
`type.split(".")[0]` yields `C3`/`N7` and no atom mapping ever matches. Read the
element from the atom *name* instead.

**3. Old-style prmtop.** `tleap` writes no `%FLAG ATOMIC_NUMBER`, so sander
guesses the heme iron's element as −1 and aborts with
`UNSUPPORTED ELEMENT`. Round-trip the prmtop through `parmed` to write the
modern format.

**4. `MASS FE 55.0` means manganese.** The heme prep file's mass is closer to Mn
(54.938) than Fe (55.845), so element inference from mass silently assigns
**manganese**. The run then converges cleanly, reports no error, and is
chemically meaningless. Correct the mass to 55.845 *and* set the atomic number
explicitly by atom name.

## Chemistry constraint

With the iron correct, sander reports an odd electron count: Fe(III) heme is d⁵
high-spin and **sander's semiempirical QM/MM is closed-shell only**. We model the
ferrous state at net charge −2 (Fe²⁺ + porphyrin²⁻ + 2 propionate⁻), which is
self-consistent with the Fe(II) pentacoordinated parameter set. That is a real
approximation: P450 resting state and type-II complexes are ferric.

**No DFT path in this build** — sander was compiled without XTB and without
DFTB+, and QUICK/TeraChem are absent. PM6 is the only Hamiltonian here with
iron; PM3/AM1/RM1/MNDO lack it. Cheap DFT would need psi4, PySCF or ORCA.

## Cost, and why it did not finish

PM6 on 115 QM atoms is roughly 4× the cost of the 73-atom version, and 60
concurrent jobs each got ~38% of a core through memory-bandwidth contention.
100 cycles projects to 3–5 hours per pose. Next time: fewer concurrent jobs, or
a smaller QM region — ligand + porphyrin core + Fe, dropping the propionates and
vinyls.

Heme parameters are in `structure/params/heme/`. Use the **pentacoordinated**
set, whose 43 heavy-atom names match Boltz's heme exactly and which has no axial
water; `hem_water`/HEW carries `OH`/`HO1`/`HO2` that clash with the ligand.
