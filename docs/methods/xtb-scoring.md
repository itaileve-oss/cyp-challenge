# GFN2-xTB interaction energy

Semi-empirical quantum interaction energy on a truncated active site. Fast,
iron-capable, and — as implemented here — too truncated to be trusted.

## Why xTB rather than a force field or an organic-only MLIP

Half the ligands bind through an Fe–N dative bond, and most cheap scorers cannot
represent it:

- **GAFF / OpenFF / MM-GBSA** have no usable Fe coordination parameters
- **Egret-1** is organic-element only
- **Vina-type scoring** reads the bond as a clash (0/20 coordinating poses)

GFN2-xTB is semi-empirical QM and handles transition metals, so the bond is
described rather than punished. (**OrbMol is also iron-capable**, via OMol25
which includes metal complexes — a better choice for pocket-scale work, and the
one we would use next time.)

## Method

```bash
python structure/src/xtb_score.py POSE_CSV OUT_CSV --valid-only --workers 90
```

Three single points per pose at the pose geometry, no relaxation:

```
ΔE = E(ligand + heme + Cys442) − E(heme + Cys442) − E(ligand)
```

GFN2 with ALPB water, charge −1 and `uhf 5` (Fe(III) high spin). 68 atoms,
~1.3 s per single point. 3,020 poses scored in ~5 min on 7 cores.

**ΔE is comparable only within a ligand.** Charge/spin assumptions and ligand
composition differ between ligands, so cross-ligand comparison is meaningless.
Within-ligand ranking is all that is needed to pick 1 of N.

## The flaw

The model is ligand + heme + Cys442 — **no protein**. So ΔE measures the
ligand–heme interaction alone and is blind to whether the rest of the molecule
fits the pocket, which biases it toward poses lying flat against the porphyrin.

Evidence: the lowest-ΔE pose coincided with the cluster medoid for only 1 of 20
ligands, and fell inside the modal cluster 11/20 times — no better than the
~14/20 expected by chance at 73% occupancy. The two signals are near-orthogonal,
and the truncation is the likely reason.

## Why we did not simply enlarge the region

Measured, so it need not be retried:

| system | atoms | GFN2 result |
|---|---|---|
| ligand + heme + Cys | 68 | **1.3 s** |
| 6 Å pocket | 954 | **no energy in 25 min** |
| whole protein (GFN-FF) | 8,280 | **no energy in 13 min** |

Pocket-scale GFN2 is out, and so is GFN2 relaxation, since an optimisation costs
50–200 single points. That is what pushed us to
[QM/MM](qmmm-minimization.md), where the QM region stays small and the pocket is
handled classically.

## Operational notes

- **xtb segfaults on large systems** unless `ulimit -s unlimited` and
  `OMP_STACKSIZE=4G` are set.
- **Run single-threaded.** Eight threads on a 70-atom system is pure overhead:
  the same single point took 5.1 s wall on 8 threads and ~1.3 s on one.
