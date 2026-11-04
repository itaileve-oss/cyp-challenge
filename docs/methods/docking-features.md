# Docking and pharmacophore features for the regression track

Structure-derived descriptors added to the ligand-only ensemble, on the premise
that a ligand-only model recalls the activity of similar training compounds and
fails on activity cliffs those compounds cannot explain.

## Pipeline

```bash
python src/prepare_cyp3a4_receptor.py            # receptor prep
python src/prepare_ligands_gd2heme_v2.py CYP3A4  # ligand mol2, protonated + stereo-enumerated
bash  src/launch_docking_jobs.sh                 # GalaxyDock2-HEME across a receptor ensemble
python src/parse_docking_results.py              # scores -> features
```

Docking uses [GalaxyDock2-HEME](galaxydock2-heme.md) against a 4–5 receptor
ensemble per isoform, aligned into a common frame so one box centre serves all.
`src/run_batch_ensemble.py` is the parallel driver, parameterised by receptor,
heme residue number, chain and box centre.

## Ligand preparation differs from the structure track

Here ligands **are** protonated (dimorphite-dl at pH 7.4) and undefined
stereocentres **are** enumerated — 926 of 2,335 CYP3A4 training SMILES have at
least one, a real ambiguity in the source data. Full enumeration expands 2,335
compounds to 3,825 variants, cheap enough to dock all of them, with a manifest
mapping variants back to compounds so aggregation can compare *best-of-variants*
against *mean-of-variants* empirically rather than by assumption.

This is the opposite choice from the structure track, where the submission
validator checks connectivity against the verbatim SMILES and protonating an
azole would remove the coordinating lone pair. The difference is deliberate: a
feature generator may alter the molecule, a submitted pose may not.

## Pharmacophores, per isoform

- **CYP3A4** — Kaur-style distances measured *from the heme-coordinating atom*:
  aromatic, amide, hydrophobic and terminal group offsets.
- **CYP1A2** — four points: flat aromatic ring, hydrogen-bond acceptor,
  hydrophobe, and a hydrogen-bond **donor**. The donor is beyond the literature's
  three-point description and is what made the feature set work.
- **CYP2D6** — basic-amine to aromatic-ring distance, the classic 2D6 motif.

## Aggregation

Docking scores are aggregated across the receptor ensemble either as the best
score or as a per-receptor vector. CYP2D6 gained from best-across-receptors;
CYP2C9 gained from neither, under either strategy.

## Caveat carried from the structure track

Vina-type coordination scoring is unreliable on a heme centre: only ~3% of
Vina-docked poses coordinate the iron against ~83% of real inhibitors. The
features are still useful as relative shape and complementarity signals, but a
docking score should not be read as evidence about coordination. This is why the
regression docking runs use GD2-HEME rather than a generic engine.
