# Frame-corrected pose clustering

Groups poses by binding mode. The important part is the correction it applies
first, without which every downstream comparison is meaningless.

## The frame problem

**Boltz emits every diffusion sample in its own arbitrary rigid-body frame.**
Measured across one ligand's samples:

| comparison | CA RMSD |
|---|---|
| raw, no superposition | **34.5 Å** |
| after Kabsch fit | **1.40 Å** |

So comparing ligand coordinates directly measures the coordinate frame, not the
pose. Before the correction, apparent ligand diversity was ~8 Å and consensus
scoring was noise. After it:

| | before | after |
|---|---|---|
| mean pairwise ligand RMSD | 5.5–10.1 Å | **2.14 Å** |
| top-cluster occupancy | — | **73%** |

Every pose pick changed. The ensemble is far more converged than it first looked.

## Method

```bash
python structure/src/cluster_poses.py ANALYSIS_DIR 2.0
```

1. Filter to PoseBusters-valid poses; apply the Fe prior where the ligand has an
   accessible aromatic sp² nitrogen.
2. Superpose each pose onto a reference sample by **CA atoms**, and carry the
   ligand through that transform.
3. Compute **symmetry-aware RMSD in place** (`rdMolAlign.CalcRMS`) — never
   `GetBestRMS`, which would align the ligands to each other and hide the
   translation that distinguishes binding modes.
4. Butina leader clustering at a 2 Å cutoff, the conventional "same pose"
   threshold.

## Two implementation traps

**Pose names collide across seeds.** Every pass writes `<MOL>_model_0..19`, so
`path.stem` is not unique. Merging on it multiplied rows ninefold and silently
overwrote per-pose files, so scores were attributed to whichever seed wrote
last. Prefix the seed directory: `full_seed2_<MOL>_model_0`.

**Aromaticity perception differs by source.** A ligand read from a GD2 mol2 and
the same ligand from a Boltz PDB may not substructure-match, so `CalcRMS` throws
even with identical atom counts. Fall back to comparing flattened topology (all
bonds single, aromatic flags cleared), which stays symmetry-aware through graph
automorphism and only slightly understates RMSD.

## Do not select on the cluster mode

Clustering is useful for *describing* the pool. As a selection rule it does not
transfer: picking the largest cluster's medoid finished below the trivial
baseline of taking the highest-confidence valid sample, and another entrant
measured consensus/medoid selection losing 0.021 on the blind set while looking
good on deposited structures. See [pose-selection.md](pose-selection.md).
