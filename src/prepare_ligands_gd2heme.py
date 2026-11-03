"""Regenerate correctly-protonated ligand mol2 files for GalaxyDock2-HEME docking.

Replaces the previous `cyp3a4_all_mol2` set, which was found (2026-09-18) to have
lost ionization-state information in whatever PDBQT->MOL2 conversion produced it:
Tripos MOL2's ATOM record has no formal-charge field, so an ionized atom with an
incomplete valence (e.g. a phenolate O with only 1 bond, no H) silently looked like
a parsing error to GalaxyDock2-HEME's chimera_proc_ligand.py step, which
unconditionally runs `del H` then `addh` (Chimera's generic, pH-UNAWARE
valence-completer) on every ligand -- so it "repaired" these atoms back to their
neutral form, undoing the intended ionization.

Fix, verified empirically against chimera_proc_ligand.py before this script was
run at scale: build from the canonical training SMILES via
  1. dimorphite_dl at pH 7.4 (single point, top variant -- same convention as
     prepare_fit_ligands.py) for the ionization state,
  2. RDKit ETKDGv3 + MMFF single-conformer 3D embed,
  3. obabel SDF->MOL2 (not PDBQT->MOL2) -- obabel's own atom-typer correctly
     assigns special types like O.co2 for carboxylate/phenolate oxygens and writes
     an explicit UNITY_ATOM_ATTR charge block, which chimera_proc_ligand.py's
     del-H/addh step DOES respect (verified: the anion survived intact).

Known remaining limitation, NOT fixed here: 926/2335 (39.7%) of CYP3A4 training
SMILES have at least one undefined stereocenter. RDKit's embedder still resolves
these to some definite 3D arrangement (an arbitrary single enantiomer) since 3D
coordinates cannot represent "unspecified" -- this is a real ambiguity in the
source dataset (the true tested stereoisomer isn't recorded), not something this
script's fix addresses.
"""

import multiprocessing as mp
import subprocess as sp
from pathlib import Path

import dimorphite_dl
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv"
OUT_DIR = PROJECT_ROOT / "data" / "structures" / "cyp3a4_all_mol2_v2"
OBABEL = "/home/ileven/miniconda3/envs/GNINA/bin/obabel"

DEFAULT_PH = 7.4
SEED = 42


def protonate_dominant(smiles: str, ph: float = DEFAULT_PH) -> str | None:
    try:
        variants = dimorphite_dl.protonate_smiles(smiles, ph_min=ph, ph_max=ph)
    except Exception:
        return None
    return variants[0] if variants else None


def prepare_one(args) -> tuple[str, bool, str]:
    name, smiles = args
    out_mol2 = OUT_DIR / f"{name}.mol2"
    try:
        prot_smi = protonate_dominant(smiles)
        if prot_smi is None:
            return (name, False, "dimorphite failed")
        mol = Chem.MolFromSmiles(prot_smi)
        if mol is None:
            return (name, False, "MolFromSmiles failed on protonated SMILES")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        params.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id < 0:
            return (name, False, "embed failed")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)

        sdf_path = out_mol2.with_suffix(".sdf")
        Chem.MolToMolFile(mol, str(sdf_path))

        result = sp.run(
            [OBABEL, str(sdf_path), "-O", str(out_mol2), "--partialcharge", "gasteiger"],
            capture_output=True, text=True, timeout=60,
        )
        sdf_path.unlink(missing_ok=True)
        if not out_mol2.exists():
            return (name, False, f"obabel failed: {result.stderr[-300:]}")
        return (name, True, "")
    except Exception as e:
        return (name, False, str(e))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_CSV)
    train = train.dropna(subset=["CYP3A4_pIC50_direct_inhibition"])[["Molecule_Name", "SMILES"]]
    print(f"{len(train)} CYP3A4 ligands to regenerate", flush=True)

    jobs = list(train.itertuples(index=False, name=None))
    n_ok = 0
    failures = []
    with mp.Pool(16) as pool:
        for i, (name, ok, msg) in enumerate(pool.imap_unordered(prepare_one, jobs), 1):
            if ok:
                n_ok += 1
            else:
                failures.append((name, msg))
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] ok={n_ok} failed={len(failures)}", flush=True)

    print(f"\nDONE: {n_ok}/{len(jobs)} succeeded")
    if failures:
        print(f"Failures ({len(failures)}):")
        for name, msg in failures[:20]:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
