# Method documentation

One document per method, with how to run it, what it produced, and where it
fails.

## Structure track

| | |
|---|---|
| [Boltz-2 co-folding](boltz2-cofolding.md) | pose generation; why plain defaults beat every tuned alternative |
| [GalaxyDock2-HEME](galaxydock2-heme.md) | heme-aware docking; right distances, wrong approach angle |
| [GNINA](gnina-docking.md) | negative control — 0/20 coordinating poses |
| [PoseBusters validity gate](posebusters-validation.md) | zero-tolerance validity, plus the stereochemistry check nothing else does |
| [Frame-corrected clustering](pose-clustering.md) | the 34.5 Å → 1.40 Å frame correction every comparison needs |
| [Crystal contact motif](crystal-motif.md) | knowledge-based scoring from 96 deposited complexes, and why it misled us |
| [GFN2-xTB scoring](xtb-scoring.md) | quantum interaction energy, and the measured limits of scaling it |
| [QM/MM minimisation](qmmm-minimization.md) | sander PM6; four silent failures worth knowing about |
| [OrbMol MLIP](orbmol.md) | ωB97M-level energies/forces for the heme site; Fe–N 2.19 Å where PM6 gave 1.96 |
| [Pose selection](pose-selection.md) | what does not transfer, and the rules that do |

## Regression track

| | |
|---|---|
| [Ensemble](regression-ensemble.md) | multitask base plus per-isoform blends |
| [Docking and pharmacophore features](docking-features.md) | structure-derived descriptors |
| [TabPFN blend](tabpfn-blend.md) | adopting a model that is worse than the one you have |
| [Placement calibration](placement-calibration.md) | fitting spread to the metric, and the location offset we never measured |

## If you read only one thing

[pose-selection.md](pose-selection.md) for the structure track and
[placement-calibration.md](placement-calibration.md) for the regression track.
Both describe where the largest available gains were, and in both cases we found
them after submitting rather than before.
