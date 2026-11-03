"""Per-compound docking features from the 5-receptor ensemble + heme-coordination
pose re-ranking (models/gd2heme_reranked_raw.csv, see src/rerank_docking_poses.py).

Aggregation: best-of-stereo (min coord_energy) per receptor, then mean across the
5 receptor conformations -- the aggregation strategy established (in the earlier
per-receptor-vs-ensemble comparison) to correlate best with true CYP3A4 pIC50.
Uses the coord_first-reranked pose (prioritizes real heme-Fe coordination
geometry over GalaxyDock2's own top-energy pick where the two disagree), the
best-performing re-ranking strategy tested so far (Fe-coordination 79.8% vs 34.7%
for plain rank-1, Pearson r -0.365 vs -0.358).

CYP3A4-only: docking was only run for the 2335 CYP3A4-labeled training compounds
(2334 successfully; missing compounds are mean-filled). Do not stack this onto the
shared multitask model directly -- see the earlier pharmacophore-feature lesson:
a feature real for only a CYP3A4-sized row subset perturbs the shared tree
ensemble's global structure even when zeroed elsewhere. Test standalone on an
isolated CYP3A4 single-task model first.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RERANKED_RAW = PROJECT_ROOT / "models" / "gd2heme_reranked_raw.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv"

FEATURE_COLS = ["coord_energy", "coord_fe_dist", "coord_frac"]


def _build_per_compound_table() -> pd.DataFrame:
    raw = pd.read_csv(RERANKED_RAW)
    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    raw["coord_flag"] = raw["coord_fe_dist"] < 3.5

    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).coord_energy.idxmin()]
    per_compound = best_per_receptor.groupby("Molecule_Name", as_index=False).agg(
        coord_energy=("coord_energy", "mean"),
        coord_fe_dist=("coord_fe_dist", "mean"),
        coord_frac=("coord_flag", "mean"),
    )
    return per_compound


def docking_feature_matrix(molecule_names: list[str]) -> np.ndarray:
    """Returns (len(molecule_names), 3) array: [coord_energy, coord_fe_dist,
    coord_frac]. Missing compounds (not docked) are filled with the column mean
    and won't be distinguishable from a real "average" compound -- fine here
    since every consumer masks rows by CYP3A4 label availability, and only 1 of
    2335 CYP3A4-labeled compounds is missing."""
    per_compound = _build_per_compound_table()
    lookup = per_compound.set_index("Molecule_Name")[FEATURE_COLS]
    means = lookup.mean()

    rows = []
    n_missing = 0
    for name in molecule_names:
        if name in lookup.index:
            rows.append(lookup.loc[name].to_numpy(dtype=float))
        else:
            rows.append(means.to_numpy(dtype=float))
            n_missing += 1
    if n_missing:
        print(f"  [features_docking] {n_missing}/{len(molecule_names)} compounds missing docking data, mean-filled")
    return np.array(rows, dtype=float)
