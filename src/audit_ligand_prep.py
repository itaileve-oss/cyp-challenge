"""Audit whether the ligand mol2 files fed into GalaxyDock2-HEME docking have
correct protonation states, and whether stereochemistry is well-defined.

Motivation: chimera_proc_ligand.py (the vendor ligand-prep step inside
GalaxyDock2-HEME) unconditionally runs `del H` then `addh` on every ligand --
it deletes any hydrogens we supply and re-adds them using Chimera's generic
valence-based heuristic, which has NO pH/pKa awareness. If our mol2 inputs
don't already carry the correct FORMAL CHARGE on ionizable atoms (not just
explicit H), Chimera's addh will silently default most groups to their neutral
form regardless of physiological pH.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

TRAIN_CSV = Path("/home/ileven/Projects/cyp-challenge/data/raw/TRAIN_inhibition.csv")
MOL2_DIR = Path("/home/ileven/Projects/cyp-challenge/data/structures/cyp3a4_all_mol2")

# SMARTS for groups that are substantially ionized at pH 7.4 (approximate, same
# spirit as dimorphite_dl's rule set).
IONIZABLE_PATTERNS = {
    "carboxylic_acid (should be COO-)": "[CX3](=O)[OX2H1]",
    "sulfonic_acid (should be SO3-)": "[SX4](=O)(=O)[OX2H1]",
    "phosphate_OH (should be deprotonated)": "[PX4](=O)([OX2H1])",
    "tetrazole (should be anionic)": "c1nnn[nH]1",
    "primary/secondary/tertiary aliphatic amine (should be protonated)": "[NX3;H2,H1,H0;!$(NC=O);!$(N=*);!$(Nc)]",
    "guanidine (should stay protonated)": "[NX3][CX3](=[NX2])[NX3]",
}


def count_pattern_hits(mol: Chem.Mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return -1
    return len(mol.GetSubstructMatches(patt))


def check_mol2_charge_state(mol2_path: Path) -> dict:
    """Does this ligand's mol2 already carry non-zero formal charges (i.e. was
    an ionization decision actually encoded before Chimera's del-H/addh)?"""
    mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False)
    if mol is None:
        return {"parse_ok": False}
    total_formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    n_explicit_h = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
    n_heavy = sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "H")
    return {
        "parse_ok": True,
        "total_formal_charge": total_formal_charge,
        "n_explicit_h": n_explicit_h,
        "n_heavy_atoms": n_heavy,
    }


def main():
    train = pd.read_csv(TRAIN_CSV)
    train = train.dropna(subset=["CYP3A4_pIC50_direct_inhibition"])[["Molecule_Name", "SMILES"]]
    print(f"{len(train)} CYP3A4-labeled training molecules")

    # --- Part 1: how many molecules carry an ionizable group at all? ---
    counts = {k: 0 for k in IONIZABLE_PATTERNS}
    unassigned_stereo_count = 0
    parse_fail = 0
    for _, row in train.iterrows():
        mol = Chem.MolFromSmiles(row.SMILES)
        if mol is None:
            parse_fail += 1
            continue
        for label, smarts in IONIZABLE_PATTERNS.items():
            if count_pattern_hits(mol, smarts) > 0:
                counts[label] += 1
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        if any(c[1] == "?" for c in centers):
            unassigned_stereo_count += 1

    print(f"\nSMILES parse failures: {parse_fail}")
    print("\n--- Ionizable-group prevalence in the CYP3A4 training set (from raw SMILES) ---")
    for label, n in counts.items():
        print(f"  {label}: {n} molecules ({100*n/len(train):.1f}%)")
    print(f"\n--- Stereochemistry ---")
    print(f"  Molecules with >=1 UNDEFINED stereocenter in the raw SMILES: {unassigned_stereo_count} "
          f"({100*unassigned_stereo_count/len(train):.1f}%)")

    # --- Part 2: what protonation state actually made it into the mol2 files
    # that were fed to GalaxyDock2-HEME? ---
    print("\n--- Actual mol2 files fed to docking: formal charge / explicit-H audit (sample of 200) ---")
    sample = train.sample(n=min(200, len(train)), random_state=0)
    zero_charge = 0
    checked = 0
    total_h_missing_flag = 0
    for _, row in sample.iterrows():
        mol2_path = MOL2_DIR / f"{row.Molecule_Name}.mol2"
        if not mol2_path.exists():
            continue
        info = check_mol2_charge_state(mol2_path)
        if not info.get("parse_ok"):
            continue
        checked += 1
        if info["total_formal_charge"] == 0:
            zero_charge += 1
    print(f"  Checked {checked} mol2 files")
    print(f"  Files with net formal charge == 0 (no ionization encoded at all): {zero_charge} "
          f"({100*zero_charge/checked:.1f}%)")


if __name__ == "__main__":
    main()
