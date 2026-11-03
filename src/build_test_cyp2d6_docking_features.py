"""Rerank + best-across-receptors docking(3) feature build for TEST_BLINDED's
CYP2D6 docking (5 receptors). Mirrors build_test_cyp3a4_docking_features.py,
using the confirmed-best aggregation (best-across-receptors, not mean) and
coordination-first winner selection for pose extraction.

Output:
  models/test_cyp2d6_gd2heme_reranked_raw.csv
  models/test_cyp2d6_docking_best.csv          -- per-molecule coord_energy/coord_fe_dist/coord_frac
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rerank_docking_poses_general import parse_fe_coord, process_ligand  # noqa: E402

RUNS_ROOT = PROJECT_ROOT / "data" / "docking_results"
MANIFEST_CSV = PROJECT_ROOT / "data" / "structures" / "test_blinded_variant_manifest.csv"

CYP2D6_RECEPTORS = {
    r: PROJECT_ROOT / f"data/receptors/cyp2d6/{r}_prepared.pdb"
    for r in ["3qm4", "3tbg", "4wnu", "4xrz", "5tft"]
}


def parse_receptor(receptor: str, receptor_pdb: Path) -> pd.DataFrame:
    fe = parse_fe_coord(receptor_pdb)
    run_root = RUNS_ROOT / f"gd2heme_runs_test_cyp2d6_{receptor}"
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


def main() -> None:
    print("Re-ranking CYP2D6 TEST_BLINDED docking (5 receptors)...", flush=True)
    dfs = []
    for receptor, pdb in CYP2D6_RECEPTORS.items():
        df_r = parse_receptor(receptor, pdb)
        print(f"  [{receptor}] {len(df_r)} ligand variants done", flush=True)
        dfs.append(df_r)
    raw = pd.concat(dfs, ignore_index=True)

    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw.to_csv(PROJECT_ROOT / "models" / "test_cyp2d6_gd2heme_reranked_raw.csv", index=False)
    print(f"\nTotal: {len(raw)} rows", flush=True)

    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5
    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    best_overall = best_per_receptor.loc[best_per_receptor.groupby("Molecule_Name").coord_energy.idxmin()]
    agg = best_overall.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "first"), coord_fe_dist=("coord_fe_dist", "first"))
    coord_frac = best_per_receptor.groupby("Molecule_Name", as_index=False)["coord_flag"].max()
    coord_frac = coord_frac.rename(columns={"coord_flag": "coord_frac"})
    agg = agg.merge(coord_frac, on="Molecule_Name", how="left")
    agg.to_csv(PROJECT_ROOT / "models" / "test_cyp2d6_docking_best.csv", index=False)
    print(f"Saved per-molecule docking(best-across-receptors) features for {len(agg)} molecules")


if __name__ == "__main__":
    main()
