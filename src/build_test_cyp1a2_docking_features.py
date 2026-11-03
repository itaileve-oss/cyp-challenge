"""Rerank + docking(3) feature build + winning-pose extraction for
TEST_BLINDED's CYP1A2 docking (1 receptor). Mirrors
build_test_cyp3a4_docking_features.py / build_test_cyp2d6_docking_features.py.

Output:
  models/test_cyp1a2_gd2heme_reranked_raw.csv
  models/test_cyp1a2_docking.csv                -- per-molecule coord_energy/coord_fe_dist/coord_frac
  models/test_cyp1a2_winning_docked_poses.pkl    -- for the pharmacophore v3 feature
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank_docking_poses_general import parse_bank_energies, parse_fe_coord, process_ligand  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "data" / "docking_results"
MANIFEST_CSV = PROJECT_ROOT / "data" / "structures" / "test_blinded_variant_manifest.csv"
CYP1A2_RECEPTOR_PDB = PROJECT_ROOT / "data/receptors/cyp1a2/2hi4_prepared.pdb"


def extract_pose(run_dir: Path, target_energy: float) -> dict | None:
    einfo = run_dir / "GD2_HEME_fb.E.info"
    mol2 = run_dir / "GD2_HEME_fb.mol2"
    if not einfo.exists() or not mol2.exists():
        return None
    energies = np.array(parse_bank_energies(einfo))
    energies = np.where(np.isnan(energies), np.inf, energies)
    text = mol2.read_text()
    blocks = text.split("@<TRIPOS>MOLECULE")[1:]
    n = min(len(energies), len(blocks))
    if n == 0:
        return None
    idx = int(np.argmin(np.abs(energies[:n] - target_energy)))
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
    fe = parse_fe_coord(CYP1A2_RECEPTOR_PDB)
    run_root = RUNS_ROOT / "gd2heme_runs_test_cyp1a2_2hi4"
    run_dirs = sorted(run_root.iterdir())
    rows = []
    for i, rd in enumerate(run_dirs, 1):
        if not rd.is_dir():
            continue
        r = process_ligand(rd, fe)
        if r is not None:
            r["receptor"] = "2hi4"
            rows.append(r)
        if i % 500 == 0:
            print(f"  {i}/{len(run_dirs)}", flush=True)
    raw = pd.DataFrame(rows)

    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw.to_csv(PROJECT_ROOT / "models" / "test_cyp1a2_gd2heme_reranked_raw.csv", index=False)
    print(f"Total: {len(raw)} rows", flush=True)

    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5
    agg = raw.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "min"), coord_fe_dist=("coord_fe_dist", "min"),
        coord_frac=("coord_flag", "max"))
    agg.to_csv(PROJECT_ROOT / "models" / "test_cyp1a2_docking.csv", index=False)
    print(f"Saved per-molecule docking features for {len(agg)} molecules", flush=True)

    print("Extracting winning poses (coordination-first)...", flush=True)
    can_coord = raw.groupby("Molecule_Name")["coord_flag"].transform("any")
    pool = raw[~can_coord | raw["coord_flag"]]
    winner_idx = pool.groupby("Molecule_Name")["coord_energy"].idxmin()
    winners = raw.loc[winner_idx, ["Molecule_Name", "variant_id", "coord_energy"]]
    out = {}
    n_err = 0
    for row in winners.itertuples(index=False):
        rd = run_root / row.variant_id
        pose = extract_pose(rd, row.coord_energy)
        if pose is None:
            n_err += 1
            continue
        pose["receptor"] = "2hi4"
        pose["energy"] = row.coord_energy
        out[row.Molecule_Name] = pose
    print(f"Extracted {len(out)}/{len(winners)} poses ({n_err} errors)", flush=True)
    with open(PROJECT_ROOT / "models" / "test_cyp1a2_winning_docked_poses.pkl", "wb") as f:
        pickle.dump(out, f)
    print("Saved winning poses")


if __name__ == "__main__":
    main()
