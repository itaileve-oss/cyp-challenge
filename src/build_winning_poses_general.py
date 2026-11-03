"""Generalized winning-pose extractor: pulls the real docked winning pose
(lowest coord_energy across all receptors x stereo variants) for every
TRAINING compound of a given isoform. Same logic as
build_cyp2d6_winning_poses.py, parametrized so it also covers CYP1A2/CYP2C9.

Output: models/{isoform}_winning_docked_poses.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank_docking_poses_general import parse_bank_energies, parse_fe_coord  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "data" / "docking_results"

ISOFORM_RECEPTORS = {
    "CYP1A2": {"2hi4": PROJECT_ROOT / "data/receptors/cyp1a2/2hi4_prepared.pdb"},
    "CYP2C9": {
        r: PROJECT_ROOT / f"data/receptors/cyp2c9/{r}_prepared.pdb"
        for r in ["1r9o", "4nz2", "5a5i", "5k7k", "5w0c"]
    },
    "CYP2D6": {
        r: PROJECT_ROOT / f"data/receptors/cyp2d6/{r}_prepared.pdb"
        for r in ["3qm4", "3tbg", "4wnu", "4xrz", "5tft"]
    },
}


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
    isoform = sys.argv[1] if len(sys.argv) > 1 else "CYP1A2"
    receptors = ISOFORM_RECEPTORS[isoform]
    manifest_csv = PROJECT_ROOT / "data" / "structures" / f"{isoform.lower()}_variant_manifest.csv"
    reranked_csv = PROJECT_ROOT / "models" / f"{isoform.lower()}_gd2heme_reranked_raw.csv"

    raw = pd.read_csv(reranked_csv)
    manifest = pd.read_csv(manifest_csv)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5

    print("Selecting winning (receptor, variant) pose per molecule "
          "(coordination-first across receptors, energy as tiebreak/fallback)...", flush=True)
    can_coord = raw.groupby("Molecule_Name")["coord_flag"].transform("any")
    pool = raw[~can_coord | raw["coord_flag"]]  # coordinating rows only, when any exist
    winner_idx = pool.groupby("Molecule_Name")["coord_energy"].idxmin()
    winners = raw.loc[winner_idx, ["Molecule_Name", "variant_id", "receptor", "coord_energy"]]
    n_coord = raw.loc[winner_idx, "coord_flag"].sum()
    print(f"  {len(winners)} molecules with a winning pose "
          f"({n_coord}/{len(winners)} = {100*n_coord/len(winners):.1f}% actually coordinate)", flush=True)

    fe_coords = {r: parse_fe_coord(p) for r, p in receptors.items()}
    out = {}
    n_err = 0
    for i, row in enumerate(winners.itertuples(index=False), 1):
        run_dir = RUNS_ROOT / f"gd2heme_runs_{isoform.lower()}_{row.receptor}" / row.variant_id
        try:
            pose = extract_pose(run_dir, row.coord_energy)
            if pose is None:
                n_err += 1
                continue
            pose["receptor"] = row.receptor
            pose["energy"] = row.coord_energy
            out[row.Molecule_Name] = pose
        except Exception as e:
            n_err += 1
            print(f"error on {row.Molecule_Name}: {e}", flush=True)
        if i % 200 == 0:
            print(f"  [{i}/{len(winners)}]", flush=True)

    print(f"\nExtracted {len(out)}/{len(winners)} poses ({n_err} errors)", flush=True)
    out_path = PROJECT_ROOT / "models" / f"{isoform.lower()}_winning_docked_poses.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
