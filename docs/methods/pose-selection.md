# Pose selection

How to choose one pose from thousands. This is the hardest part of the task and
the part we got wrong, so this document is mostly a record of what does not work.

## The problem

Boltz confidence is nearly flat across samples — ipTM 0.946–0.990 across all
3,020 valid poses — while those samples sit several Å apart. So "take the most
confident" is close to arbitrary, and nothing in the model's own output ranks
them.

## What we tried, and the outcome

| criterion | implementation | result |
|---|---|---|
| largest-cluster medoid | `cluster_poses.py` | **below baseline** |
| crystal-motif recall | `motif_score.py` | **below baseline** |
| xTB interaction energy | `xtb_score.py` | truncated model, near-orthogonal to consensus |
| rank fusion of the above | `select_final.py` | degenerate — ties in cluster population let ΔE dominate |

Our submitted entry (16 motif-selected, 2 medoid, 2 docked) scored **LDDT-PLI
0.371**. Another entrant's plain top-confidence valid pose scores **0.414**. Our
entire selection stack performed worse than the trivial baseline.

The three criteria also disagreed with each other at roughly chance level —
cluster-medoid and lowest-ΔE picks shared 0 of 20 ligands, and the lowest-ΔE pose
fell in the modal cluster 11/20 times against ~14/20 expected by chance. That
pattern should have been read as "none of these tracks accuracy", not as a reason
to fuse them.

## What does work

Published, leaderboard-verified rules from another entrant, whose live score rose
from 0.414 to 0.484 across these steps:

**a. Base.** The highest-confidence sample that passes the validity gate, for
every ligand. → 0.414

**b. Far orientation, iron-bound.** For ligands that coordinate the heme iron,
take the valid sample *furthest* from the base pose that is still bound to the
iron. → **0.454**

```bash
# our implementation of rule b
python structure/src/select_poses.py   # see also the rule-a/b driver in docs
```

**c. Bounded conformation change.** Keep position and orientation nearly fixed
(centroid ≤ 0.8 Å, rotation ≤ 15°) and take the sample with the largest change in
*internal* ligand conformation. → 0.476

**d. Cross-model agreement.** Find the pose most samples of a *second* co-folding
model agree on, and switch to the nearest sample of the first model if it is
within 2 Å. → 0.484

## Why divergence beats consensus

The modal orientation is **biased**, not merely uncertain. Picking the centre of
the largest cluster, or maximising burial, concentrates that bias. Moving away
from the mode while holding fixed the one thing the model reliably gets right —
the iron bond, which it places at ~2.1 Å and ~175° on nearly every sample —
explores the error direction instead of compounding it.

The corollary: **iron coordination cannot be used to rank poses.** It is
essentially constant across samples, and matches deposited structures already. We
spent considerable effort on Fe-distance discrimination that was measuring a
constant.

## Lessons

1. Take top-confidence valid as the baseline, and require any selector to beat it
   on held-out data before trusting it.
2. Do not select on within-model consensus. It looks good on deposited structures
   because the model has seen them.
3. Validate a selector against held-out *crystal* poses, not against its own
   internal agreement.
4. A second co-folding model is worth more than a better scorer on one model.
5. Buy leaderboard feedback early with single-target probe submissions.
