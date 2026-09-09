import pandas as pd, numpy as np
R = "/home/ileven/Projects/cyp-challenge/data/raw/"
CYP = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

tr = pd.read_csv(R + "TRAIN_inhibition.csv")
te = pd.read_csv(R + "TEST_BLINDED.csv")
tdi = pd.read_csv(R + "TRAIN_TDI.csv")
sc = pd.read_csv(R + "single_conc_TRAIN.csv")

print("=== TRAIN_inhibition ===")
print("rows", len(tr), "unique Molecule_Name", tr.Molecule_Name.nunique(), "unique SMILES", tr.SMILES.nunique())
for c in CYP:
    col = f"{c}_pIC50_direct_inhibition"
    v = tr[col].dropna()
    print(f"  {c}: n={len(v):4d}  mean={v.mean():.2f}  min={v.min():.2f}  max={v.max():.2f}  <4: {(v<4).mean()*100:.0f}%")
ci = tr["CYP3A4_pIC50_direct_inhibition_conf_high"] - tr["CYP3A4_pIC50_direct_inhibition_conf_low"]
print("  CYP3A4 CI width: mean", round(ci.mean(),2), "median", round(ci.median(),2))
print("  rows with >=1 isoform measured:", tr[[f'{c}_pIC50_direct_inhibition' for c in CYP]].notna().any(axis=1).sum())
print("  isoforms measured per compound histogram:")
print(tr[[f'{c}_pIC50_direct_inhibition' for c in CYP]].notna().sum(axis=1).value_counts().sort_index().to_dict())

print("\n=== TRAIN_TDI ===")
print("rows", len(tdi), "unique cpd", tdi.Molecule_Name.nunique())
for c in ["CYP3A4", "CYP2D6"]:
    col = f"{c}_is_TDI"
    v = tdi[col].dropna()
    print(f"  {c}_is_TDI: n={len(v)}  positives={int(v.sum())} ({v.mean()*100:.1f}%)  nan={tdi[col].isna().sum()}")

print("\n=== single_conc ===")
print(sc.enzyme.value_counts().to_dict())

print("\n=== TEST ===")
print("rows", len(te), "unique cpd", te.Molecule_Name.nunique(), "unique SMILES", te.SMILES.nunique())

# overlap train/test by SMILES
from rdkit import Chem
def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None
te_c = set(filter(None, (canon(s) for s in te.SMILES)))
tr_c = set(filter(None, (canon(s) for s in tr.SMILES)))
tdi_c = set(filter(None, (canon(s) for s in tdi.SMILES)))
print("test canonical:", len(te_c), " overlap w/ train_inhib:", len(te_c & tr_c), " overlap w/ train_tdi:", len(te_c & tdi_c))
