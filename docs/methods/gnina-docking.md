# GNINA docking — negative control

Kept in the repository deliberately, as evidence of what a standard docking
engine cannot do on a metalloenzyme.

## Result

For OCNT-2395436, a pyridine-containing ligand whose co-folded poses coordinate
the heme iron at 2.04 Å in 20 out of 20 samples:

```
20 modes generated, best affinity -7.95 kcal/mol
coordinating modes (Fe < 2.6 Å): 0 / 20
closest approach: 2.78 Å
```

Zero coordinating poses. The failure is structural, not a tuning problem: the
Vina-type scoring function has no term for a dative bond, so the Fe···N
interaction is modelled as van der Waals plus electrostatics and a 2.0 Å contact
is simply a clash to be avoided.

## Why this matters beyond this challenge

Half the ligands in this set bind through an Fe–N bond. A scorer blind to that
interaction does not merely ignore it — it actively penalises the correct pose,
so the ranking inverts exactly where the chemistry is most informative. The same
mechanism explains why Boltz's `--use_potentials` flag drops heme coordination
from 100% to 0%: a generic repulsive potential reads the coordination bond as an
overlap.

## Running it

```bash
python structure/src/dock_pool.py ANALYSIS_DIR OUT_DIR --exhaustiveness 16
```

Follows the project's established GNINA protocol — autobox from the heme with
`--autobox_add 8`, receptor built from the co-folded prediction with the heme
retained. CNN rescoring is recorded but should be weighted lightly: data-driven
scorers were reported to invert on out-of-distribution ligands in the related
PXR challenge post-mortem.

Use [GalaxyDock2-HEME](galaxydock2-heme.md) for any real CYP docking.
