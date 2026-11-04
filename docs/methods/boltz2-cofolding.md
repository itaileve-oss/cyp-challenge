# Boltz-2 co-folding

Generates CYP3A4–ligand complexes from sequence. This produced every pose we
submitted; the docking and scoring methods exist to choose among its output.

## What it does

One YAML per ligand describing three chains — protein (UniProt P08684, 503 aa),
the heme as CCD component `HEM` in its own chain, and the ligand as SMILES.
Boltz-2 predicts the whole complex. No templates, no receptor preparation, no
conformer generation: the model builds 3D coordinates itself.

```bash
python structure/src/make_boltz_inputs.py OUT_DIR --msa cyp3a4_msa.csv
boltz predict OUT_DIR --out_dir pool_seed1 --seed 1 \
  --diffusion_samples 20 --sampling_steps 100 --recycling_steps 5 \
  --output_format pdb
```

Nine seeded passes of 20 samples gave 180 poses per ligand (3,600 total).

## Two non-obvious input rules

**The ligand SMILES is used verbatim** — never re-protonated or
re-tautomerised. Two reasons: the submission validator checks connectivity
against that exact string, and the ten azole/pyridine ligands need a *neutral*
ring nitrogen to donate its lone pair to the iron. Protonating an imidazole at
pH 7.4 removes the only interaction that matters.

**The axial cysteine is Cys442** (`...YTPFGSGPRNC(442)IGMR...`). `--bond` adds a
Cys442:SG–HEM:FE constraint, but see below: don't.

## Calibration — defaults win

Measured on one ligand, 10 samples per config, against the organizers' baseline
reference:

| config | PoseBusters pass | Fe coordination | median Fe | pairwise RMSD |
|---|---|---|---|---|
| **plain defaults (step_scale 1.5)** | **100%** | **100%** | **2.07 Å** | 7.96 Å |
| + Cys442–FE bond constraint | 90% | 100% | 2.18 Å | — |
| + `--use_potentials` | 100% | **0%** | 2.86 Å | — |
| step_scale 1.2 | 80% | 100% | 2.07 Å | 8.30 Å |
| step_scale 1.0 | 22% | 100% | 2.07 Å | 8.32 Å |

All three chemically-motivated adjustments were wrong:

1. **`--use_potentials` destroys heme coordination** — 100% → 0%. A 2.0 Å
   N–Fe dative bond reads as a steric clash to a generic repulsive potential, so
   the ligand gets pushed off the iron. It is also unnecessary, since plain
   defaults already pass PoseBusters on every pose.
2. **The explicit Cys–Fe bond constraint is mildly harmful**, lengthening the
   coordination and causing the one validity failure. Boltz already knows P450
   geometry from training.
3. **Lowering `step_scale` is a bad trade** — validity collapses from 100% to
   22% for ~4% more diversity.

`sampling_steps 100` costs nothing against 200 in validity or coordination, so
it buys twice the poses.

## Cost

Throughput scales as `diffusion_samples × sampling_steps` at roughly 8.3
sample-steps/s; `--max_parallel_samples` makes no measurable difference. Peak
VRAM 12.8 GB at 10 samples, 21–23.4 GB at 20, so a 24 GB card is the practical
minimum and a 32 GB card has comfortable headroom. About 18–25 min per pass of
20 ligands × 20 samples on a 4090 or 5090.

## Gotchas

- **OOM is silent.** Boltz skips the ligand and still exits 0. Count poses after
  every pass and backfill.
- **Triton needs a C compiler.** CUDA *runtime* images have no gcc and the first
  `boltz predict` dies with "Failed to find C compiler". `apt-get install
  build-essential`.
- **Both the heme and the ligand are named `LIG`** (chain A protein, B heme, C
  ligand). Split poses by *chain*, never by residue name, and rename the heme to
  `HEM` for submission — the validator demands exactly one `LIG` residue.
- **Reuse the merged MSA** at `boltz_results_*/msa/<id>_0.csv`, not the raw
  per-database a3m files. Same protein for all 20 ligands, so build it once.
- **Every sample comes out in its own rigid-body frame.** See
  [pose-clustering.md](pose-clustering.md) — this invalidates any direct
  coordinate comparison.
