# PoseBusters validity gate

In this challenge validity is not a soft preference. A pose that fails
PoseBusters is assigned LDDT-PLI 0, and the tolerance is zero failed checks
(`POSEBUSTERS_MAX_FAILURES = 0`), so one bad pose costs ~0.02 of the 20-target
mean. Validity is therefore a hard filter applied before any ranking.

## Running the organizers' exact path

```bash
python structure/src/pb_official.py OUT_DIR PRED_DIR [PRED_DIR ...]
```

Two details make this reproduce the scorer rather than approximate it:

1. **The heme is renamed `HEM`** before checking. Boltz names both the cofactor
   and the ligand `LIG`; at scoring time the heme is part of `pb_protein` while
   the ligand is popped out. Checking with the heme still named `LIG` tests a
   different system.
2. **The challenge SMILES is passed as `mol_true`**, which enables checks a
   `mol_true=None` call never runs. The organizers also use a more tolerant
   bond-order matcher than RDKit's default — it tries every topological match
   rather than the first — so a pose can pass theirs while failing a naive call.

Results on our pool: **3,020 of 3,600 valid (84%)**, with the only failure modes
being `minimum_distance_to_protein` and a single `bond_lengths`.

## The heme is not in the protein clash check

PoseBusters classifies atoms by their HETATM flag, not residue name, so the heme
is treated as a cofactor and **excluded from `minimum_distance_to_protein`**.
A 2.0 Å Fe–N coordination bond is therefore never scored as a protein clash —
which is good, but it also means a genuine ligand–heme clash is invisible in the
default configuration.

To see heme clashes the heme must be relabelled as protein atoms:

```python
# rewrite heme HETATM -> ATOM with a standard residue name, then bust
if ln.startswith("HETATM") and ln[17:20].strip() == "HEM":
    ln = "ATOM  " + ln[6:17] + "GLY" + ln[20:]
```

Our submitted poses pass in both modes.

## PoseBusters does not check stereochemistry

Neither does the submission validator's `AssignBondOrdersFromTemplate`, which
compares atoms and bonds only. A pose can have the right connectivity and the
wrong configuration at a specified stereocentre and pass every automated check.
Audit it separately:

```python
Chem.AssignStereochemistryFrom3D(pose_mol)
assert Chem.MolToSmiles(pose_mol) == Chem.MolToSmiles(template)
```

Our 20 submitted poses: 16 match, 4 have no specified centres, 0 mismatch.

## Gotcha

Passing the raw PDB ligand to PoseBusters without assigning bond orders from the
template makes aromaticity perception fail, and
`non-aromatic_ring_non-flatness` then fires on most poses. That is a bug in the
caller, not a property of the poses — always template the ligand first.
