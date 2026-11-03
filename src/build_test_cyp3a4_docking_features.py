"""Consolidated pipeline for the CYP3A4 TEST_BLINDED compounds: re-rank each
ligand's 50-pose bank per receptor (coord_first rule), pick the single winning
(receptor, variant) pose per molecule (lowest coord_energy across the whole
5-receptor x stereo-variant bank -- same selection rule extract_winning_poses.py
used for training, reconstructed here since that script depended on an
undocumented ad-hoc /tmp/winning_poses.csv), then extract that winning pose's
real 3D coordinates for the pharmacophore feature and build the aggregated
docking_feature_matrix()-style columns (coord_energy/coord_fe_dist/coord_frac).

Output:
  models/test_cyp3a4_gd2heme_reranked_raw.csv  -- one row per (variant, receptor)
  models/test_cyp3a4_winning_docked_poses.pkl  -- one winning pose per Molecule_Name
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank_docking_poses_general import (  # noqa: E402
    fe_dist_for_pose,
    parse_bank_energies,
    parse_bank_mol2,
    parse_fe_coord,
    process_ligand,
)

RUNS_ROOT = PROJECT_ROOT / "data" / "docking_results"
MANIFEST_CSV = PROJECT_ROOT / "data" / "structures" / "test_blinded_variant_manifest.csv"
HETEROATOM_ELEMENTS = {"N", "O", "S"}

CYP3A4_RECEPTORS = {
    "4ny4": PROJECT_ROOT / "data/receptors/cyp3a4_receptor_prepared.pdb",
    "1w0f": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_1w0f_prepared.pdb",
    "1w0g": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_1w0g_prepared.pdb",
    "3nxu": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_3nxu_prepared.pdb",
    "2j0d": PROJECT_ROOT / "data/receptors/ensemble/cyp3a4_2j0d_prepared.pdb",
}


def parse_receptor(receptor: str, receptor_pdb: Path) -> pd.DataFrame:
    fe = parse_fe_coord(receptor_pdb)
    run_root = RUNS_ROOT / f"gd2heme_runs_test_cyp3a4_{receptor}"
    run_dirs = sorted(run_root.iterdir())
    rows = []
    for i, rd in enumerate(run_dirs, 1):
        if not rd.is_dir():
            continue
        r = process_ligand(rd, fe)
        if r is not None:
            r["receptor"] = receptor
            rows.append(r)
        if i % 500 == 0:
            print(f"  [{receptor}] {i}/{len(run_dirs)}", flush=True)
    return pd.DataFrame(rows)


def extract_pose(run_dir: Path, fe: np.ndarray, target_energy: float) -> dict | None:
    einfo = run_dir / "GD2_HEME_fb.E.info"
    mol2 = run_dir / "GD2_HEME_fb.mol2"
    energies = np.array(parse_bank_energies(einfo))
    poses_full = parse_bank_mol2(mol2)
    n = min(len(energies), len(poses_full))
    energies = energies[:n]
    idx = int(np.argmin(np.abs(energies - target_energy)))

    text = mol2.read_text()
    blocks = text.split("@<TRIPOS>MOLECULE")[1:]
    raw_block = "@<TRIPOS>MOLECULE" + blocks[idx]

    els, coords = [], []
    lines = blocks[idx].splitlines()
    start = lines.index("@<TRIPOS>ATOM") + 1
    for line in lines[start:]:
        if line.startswith("@<TRIPOS>"):
            break
        parts = line.split()
        if len(parts) < 5:
            continue
        sybyl = parts[5]
        element = sybyl.split(".")[0]
        if element == "H" or sybyl.startswith("H"):
            continue
        els.append(element)
        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
    return {"elements": els, "coords": np.array(coords), "mol2_block": raw_block}


def main() -> None:
    print("Re-ranking bank poses for CYP3A4 TEST_BLINDED (5 receptors)...", flush=True)
    dfs = []
    for receptor, pdb in CYP3A4_RECEPTORS.items():
        df_r = parse_receptor(receptor, pdb)
        print(f"  [{receptor}] {len(df_r)} ligand variants done", flush=True)
        dfs.append(df_r)
    raw = pd.concat(dfs, ignore_index=True)

    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw.to_csv(PROJECT_ROOT / "models" / "test_cyp3a4_gd2heme_reranked_raw.csv", index=False)
    print(f"\nTotal: {len(raw)} rows -- saved reranked raw CSV", flush=True)

    print("Selecting winning (receptor, variant) pose per molecule...", flush=True)
    winner_idx = raw.groupby("Molecule_Name")["coord_energy"].idxmin()
    winners = raw.loc[winner_idx, ["Molecule_Name", "variant_id", "receptor", "coord_energy"]]
    print(f"  {len(winners)} molecules with a winning pose", flush=True)

    fe_coords = {r: parse_fe_coord(p) for r, p in CYP3A4_RECEPTORS.items()}
    out = {}
    n_err = 0
    for i, row in enumerate(winners.itertuples(index=False), 1):
        run_dir = RUNS_ROOT / f"gd2heme_runs_test_cyp3a4_{row.receptor}" / row.variant_id
        try:
            pose = extract_pose(run_dir, fe_coords[row.receptor], row.coord_energy)
            pose["receptor"] = row.receptor
            pose["energy"] = row.coord_energy
            out[row.Molecule_Name] = pose
        except Exception as e:
            n_err += 1
            print(f"error on {row.Molecule_Name}: {e}", flush=True)
        if i % 200 == 0:
            print(f"  [{i}/{len(winners)}]", flush=True)

    print(f"\nExtracted {len(out)}/{len(winners)} poses ({n_err} errors)", flush=True)
    with open(PROJECT_ROOT / "models" / "test_cyp3a4_winning_docked_poses.pkl", "wb") as f:
        pickle.dump(out, f)
    print("Saved to models/test_cyp3a4_winning_docked_poses.pkl")


if __name__ == "__main__":
    main()
