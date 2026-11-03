"""Same ligand-prep pipeline as prepare_ligands_gd2heme_v2.py (protonate at
pH 7.4 via dimorphite-dl, enumerate undefined stereocenters, embed+MMFF,
convert to mol2 via obabel with Gasteiger charges) applied to the 750-compound
TEST_BLINDED.csv instead of a TRAIN_inhibition.csv isoform slice -- no
endpoint column to filter on, every row gets prepared, since we need CYP3A4
docking features for the actual blind submission (test compounds were never
docked before now).
"""

import multiprocessing as mp
import subprocess as sp
from pathlib import Path

import dimorphite_dl
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
TEST_CSV = PROJECT_ROOT / "data" / "raw" / "TEST_BLINDED.csv"
OBABEL = "/home/ileven/miniconda3/envs/GNINA/bin/obabel"
OUT_DIR = PROJECT_ROOT / "data" / "structures" / "test_blinded_all_mol2"

DEFAULT_PH = 7.4
SEED = 42


def enumerate_stereo_smiles(smiles: str) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True)
    try:
        isomers = list(EnumerateStereoisomers(mol, options=opts))
    except Exception:
        return [smiles]
    if not isomers:
        return [smiles]
    return [Chem.MolToSmiles(m) for m in isomers]


def protonate_dominant(smiles: str, ph: float = DEFAULT_PH) -> str | None:
    try:
        variants = dimorphite_dl.protonate_smiles(smiles, ph_min=ph, ph_max=ph)
    except Exception:
        return None
    return variants[0] if variants else None


def prepare_one(args) -> tuple[str, bool, str]:
    variant_id, smiles = args
    out_mol2 = OUT_DIR / f"{variant_id}.mol2"
    try:
        prot_smi = protonate_dominant(smiles)
        if prot_smi is None:
            return (variant_id, False, "dimorphite failed")
        mol = Chem.MolFromSmiles(prot_smi)
        if mol is None:
            return (variant_id, False, "MolFromSmiles failed on protonated SMILES")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        params.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, params)
        if conf_id < 0:
            return (variant_id, False, "embed failed")
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)

        sdf_path = out_mol2.with_suffix(".sdf")
        Chem.MolToMolFile(mol, str(sdf_path))

        result = sp.run(
            [OBABEL, str(sdf_path), "-O", str(out_mol2), "--partialcharge", "gasteiger"],
            capture_output=True, text=True, timeout=60,
        )
        sdf_path.unlink(missing_ok=True)
        if not out_mol2.exists():
            return (variant_id, False, f"obabel failed: {result.stderr[-300:]}")
        return (variant_id, True, "")
    except Exception as e:
        return (variant_id, False, str(e))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST_CSV)[["Molecule_Name", "SMILES"]]
    print(f"{len(test)} TEST_BLINDED compounds", flush=True)

    manifest_rows = []
    jobs = []
    for name, smiles in test.itertuples(index=False, name=None):
        stereo_smis = enumerate_stereo_smiles(smiles)
        if not stereo_smis:
            stereo_smis = [smiles]
        for i, s in enumerate(stereo_smis):
            variant_id = f"{name}__v{i}"
            jobs.append((variant_id, s))
            manifest_rows.append({"variant_id": variant_id, "Molecule_Name": name, "n_variants": len(stereo_smis)})

    print(f"{len(jobs)} total ligand variants after stereo enumeration", flush=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(PROJECT_ROOT / "data" / "structures" / "test_blinded_variant_manifest.csv", index=False)

    n_ok = 0
    failures = []
    with mp.Pool(8) as pool:
        for i, (variant_id, ok, msg) in enumerate(pool.imap_unordered(prepare_one, jobs), 1):
            if ok:
                n_ok += 1
            else:
                failures.append((variant_id, msg))
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] ok={n_ok} failed={len(failures)}", flush=True)

    print(f"\nDONE: {n_ok}/{len(jobs)} succeeded")
    if failures:
        print(f"Failures ({len(failures)}):")
        for variant_id, msg in failures[:20]:
            print(f"  {variant_id}: {msg}")


if __name__ == "__main__":
    main()
