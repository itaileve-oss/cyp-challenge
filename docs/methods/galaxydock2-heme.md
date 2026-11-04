# GalaxyDock2-HEME docking

Docking with an explicit heme term. The correct choice for a P450 and the only
docking engine here that can represent iron coordination at all.

## Why not a standard docking engine

Vina-type scoring functions have no term for a dative bond, so a 2.0 Å Fe–N
contact is scored as a steric clash. Measured directly on OCNT-2395436, whose
Boltz poses coordinate in 20/20:

| engine | coordinating poses | min Fe distance |
|---|---|---|
| GNINA (Vina-type) | **0 / 20** | 2.78 Å |
| GalaxyDock2-HEME | **47 / 50** | 2.27 Å |

GD2-HEME carries a dedicated `HM_E` heme term in its score breakdown. See
[gnina-docking.md](gnina-docking.md) for the control run.

## Running it

```bash
python structure/src/dock_gd2heme_structure.py ANALYSIS_DIR --workers 4
```

Per ligand it builds a receptor, embeds the ligand, and runs GD2-HEME, writing
50 poses to `GD2_HEME_fb.mol2` with scores in `GD2_HEME_fb.E.info`.

## Setup requirements, all of which bite

- **UCSF Chimera must be on `PATH`.** The GD2 script calls `chimera --nogui` to
  assign ligand and heme partial charges. Without it the run fails late.
- **The heme must be resname `HEM` (or `HEC`), a HETATM, and in the same chain as
  the protein.** `preprocess_protein()` filters by chain, so Boltz's chain-B heme
  has to be rewritten into chain A. We use residue 601.
- **PDB columns must be exact**: 13–16 atom name, 17 altLoc, 18–20 resName, 22
  chainID, 23–26 resSeq. An off-by-one puts `HEM` in the altLoc field and GD2
  reports the heme as missing, not as malformed.
- **Use short absolute paths.** GD2 parses fixed-length path fields and fails on
  long ones; we symlink the install to `~/g2` and run from `~/gd2r/<ligand>/`.

## Design choices

**Receptor = the ligand's own co-folded prediction**, ligand stripped, heme kept.
Docked poses then land in the same coordinate frame as that ligand's co-folded
poses, so a cross-method RMSD is directly meaningful. This is defensible here
because the predicted receptor is nearly ligand-independent (1.44 Å CA RMSD
between different ligands).

**The autobox is built from the heme**, not from the co-folded ligand, so the
search is centred on the active site without being steered toward the answer
co-folding already gave.

**Ligands are embedded from the verbatim challenge SMILES** with no protonation
change — unlike the activity-track preparation, which protonates at pH 7.4. An
imidazole protonated at that pH cannot coordinate iron.

## What it gets right, and what it does not

GD2-HEME reproduces crystal Fe–N **distances** well (2.28–2.54 Å against a
crystal median of ~2.2–2.35 Å, where co-folding sits slightly short at
2.02–2.11 Å).

But it approaches from the wrong direction. The iron sits a median **49° off the
donor nitrogen's lone-pair axis** (individual cases past 60°), against 11–15° for
co-folded poses. A nitrogen 2.4 Å from the iron but 50° off-axis is not a dative
bond. This is what you would expect from a scoring function that rewards an
attractive Fe···N term without enforcing orbital directionality — and it is the
reason no GD2 pose should be chosen on its iron distance alone.
