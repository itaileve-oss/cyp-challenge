"""Re-rank GalaxyDock2-HEME's own 50-pose bank per ligand by fit to the
empirically-measured heme-Fe coordination geometry (median 2.44A, from real
CYP3A4-bound PDB structures), instead of only trusting its own top-energy
pose (rank 1) -- which the earlier diagnosis (analyze_gd2heme_ensemble.py /
item 5f) showed under-recovers real coordination geometry.

No new docking is run: the full 50-pose bank is already on disk in
GD2_HEME_fb.mol2 (multi-model) + GD2_HEME_fb.E.info (per-pose energies, same
rank order), for every (receptor, ligand variant) combo from the 2026-09-20
ensemble run.

Two re-ranking strategies, both compared against the "rank 1" baseline:
  - coord_first: among the 50 poses, if any reaches Fe-coordination range
    (<3.5A), pick the lowest-energy one among those; else fall back to rank 1.
  - soft_penalty: pick the pose minimizing (energy + LAMBDA * fe_dist) across
    all 50 -- a mild continuous bias toward coordination geometry rather than
    a hard cutoff.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
RUNS_ROOT = PROJECT_ROOT / "tmp" / "gd2heme_ensemble_20260920"
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv"
HETEROATOM_ELEMENTS = {"N", "O", "S"}
LAMBDA = 0.3  # energy units per Angstrom of Fe-distance penalty

RECEPTOR_PDBS = {
    "4ny4": PROJECT_ROOT / "data" / "receptors" / "cyp3a4_receptor_prepared.pdb",
    "1w0f": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_1w0f_prepared.pdb",
    "1w0g": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_1w0g_prepared.pdb",
    "3nxu": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_3nxu_prepared.pdb",
    "2j0d": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_2j0d_prepared.pdb",
}


def parse_fe_coord(receptor_pdb: Path) -> np.ndarray:
    for line in receptor_pdb.read_text().splitlines():
        if line.startswith("HETATM") and line[76:78].strip() == "Fe":
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    raise ValueError(f"No Fe atom found in {receptor_pdb}")


def parse_bank_energies(einfo_path: Path) -> list[float]:
    lines = [
        l for l in einfo_path.read_text().splitlines()
        if l.strip() and not l.startswith("!") and l.split()[0].isdigit()
    ]
    return [float(l.split()[1]) for l in lines]


def parse_bank_mol2(mol2_path: Path) -> list[list[tuple[str, np.ndarray]]]:
    """Returns one heavy-atom list per pose, in bank-rank order."""
    text = mol2_path.read_text()
    blocks = text.split("@<TRIPOS>MOLECULE")[1:]
    poses = []
    for block in blocks:
        lines = block.splitlines()
        try:
            start = lines.index("@<TRIPOS>ATOM") + 1
        except ValueError:
            continue
        atoms = []
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
            xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
            atoms.append((element, xyz))
        poses.append(atoms)
    return poses


def fe_dist_for_pose(atoms: list[tuple[str, np.ndarray]], fe: np.ndarray) -> tuple[float, str | None]:
    het = [(e, xyz) for e, xyz in atoms if e in HETEROATOM_ELEMENTS]
    if het:
        e, xyz = min(het, key=lambda t: np.linalg.norm(t[1] - fe))
        return float(np.linalg.norm(xyz - fe)), e
    if not atoms:
        return float("inf"), None
    xyz = min((xyz for _, xyz in atoms), key=lambda p: np.linalg.norm(p - fe))
    return float(np.linalg.norm(xyz - fe)), None


def process_ligand(run_dir: Path, fe: np.ndarray) -> dict | None:
    einfo = run_dir / "GD2_HEME_fb.E.info"
    mol2 = run_dir / "GD2_HEME_fb.mol2"
    if not einfo.exists() or not mol2.exists():
        return None
    energies = parse_bank_energies(einfo)
    poses = parse_bank_mol2(mol2)
    n = min(len(energies), len(poses))
    if n == 0:
        return None
    energies, poses = energies[:n], poses[:n]

    fe_dists = []
    for atoms in poses:
        d, _elem = fe_dist_for_pose(atoms, fe)
        fe_dists.append(d)
    fe_dists = np.array(fe_dists)
    energies = np.array(energies)

    rank1_energy, rank1_fe = energies[0], fe_dists[0]

    coord_mask = fe_dists < 3.5
    if coord_mask.any():
        idx = np.argmin(np.where(coord_mask, energies, np.inf))
    else:
        idx = 0
    coord_energy, coord_fe = energies[idx], fe_dists[idx]

    soft_score = energies + LAMBDA * fe_dists
    idx_soft = int(np.argmin(soft_score))
    soft_energy, soft_fe = energies[idx_soft], fe_dists[idx_soft]

    return {
        "variant_id": run_dir.name,
        "rank1_energy": rank1_energy, "rank1_fe_dist": rank1_fe, "rank1_coord": rank1_fe < 3.5,
        "coord_energy": coord_energy, "coord_fe_dist": coord_fe, "coord_coord": coord_fe < 3.5,
        "soft_energy": soft_energy, "soft_fe_dist": soft_fe, "soft_coord": soft_fe < 3.5,
        "n_poses": n,
    }


def parse_receptor(receptor: str) -> pd.DataFrame:
    fe = parse_fe_coord(RECEPTOR_PDBS[receptor])
    run_dirs = sorted((RUNS_ROOT / f"gd2heme_runs_{receptor}").iterdir())
    rows = []
    for i, rd in enumerate(run_dirs, 1):
        if not rd.is_dir():
            continue
        r = process_ligand(rd, fe)
        if r is not None:
            r["receptor"] = receptor
            rows.append(r)
        if i % 1000 == 0:
            print(f"  [{receptor}] {i}/{len(run_dirs)}", flush=True)
    return pd.DataFrame(rows)


def main():
    print("Re-ranking bank poses for all 5 receptors...")
    dfs = []
    for receptor in RECEPTOR_PDBS:
        df_r = parse_receptor(receptor)
        print(f"  [{receptor}] {len(df_r)} ligand variants done")
        dfs.append(df_r)
    raw = pd.concat(dfs, ignore_index=True)
    raw.to_csv(PROJECT_ROOT / "models" / "gd2heme_reranked_raw.csv", index=False)
    print(f"\nTotal: {len(raw)} rows -- saved to models/gd2heme_reranked_raw.csv")

    n_changed_coord = (raw.coord_energy != raw.rank1_energy).sum()
    n_changed_soft = (raw.soft_energy != raw.rank1_energy).sum()
    print(f"\ncoord_first picked a non-rank1 pose for {n_changed_coord}/{len(raw)} ({n_changed_coord/len(raw):.1%})")
    print(f"soft_penalty picked a non-rank1 pose for {n_changed_soft}/{len(raw)} ({n_changed_soft/len(raw):.1%})")
    print(f"\nFe-coordination rate -- rank1: {raw.rank1_coord.mean():.3f}  coord_first: {raw.coord_coord.mean():.3f}  soft_penalty: {raw.soft_coord.mean():.3f}")

    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    train = pd.read_csv(TRAIN_CSV)[["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"]].dropna()
    raw = raw.merge(train, on="Molecule_Name", how="inner")
    TARGET = "CYP3A4_pIC50_direct_inhibition"
    print(f"\nRows with true pIC50: {len(raw)}")

    def eval_strategy(energy_col, fe_col, coord_col, label):
        # mean-of-stereo per receptor, then mean-across-receptors (established best aggregation)
        per_receptor = raw.groupby(["Molecule_Name", "receptor", TARGET], as_index=False).agg(
            energy=(energy_col, "mean"), fe=(fe_col, "mean"), coord=(coord_col, "mean"))
        agg = per_receptor.groupby(["Molecule_Name", TARGET], as_index=False).agg(
            energy=("energy", "mean"), fe=("fe", "mean"), coord=("coord", "mean"))
        r, p = pearsonr(agg.energy, agg[TARGET])
        rho, ps = spearmanr(agg.energy, agg[TARGET])
        print(f"[{label}] n={len(agg)}  Pearson={r:.4f} (p={p:.2e})  Spearman={rho:.4f}  Fe-coord frac={agg.coord.mean():.3f}")
        return {"label": label, "n": len(agg), "pearson": r, "pearson_p": p, "spearman": rho, "fe_coord_frac": agg.coord.mean()}

    print("\n=== Strategy comparison (mean-of-stereo per receptor, then mean-across-receptors) ===")
    results = [
        eval_strategy("rank1_energy", "rank1_fe_dist", "rank1_coord", "baseline (rank 1 / top GD2 energy)"),
        eval_strategy("coord_energy", "coord_fe_dist", "coord_coord", "coord_first (hard 3.5A cutoff, energy tiebreak)"),
        eval_strategy("soft_energy", "soft_fe_dist", "soft_coord", f"soft_penalty (energy + {LAMBDA}*fe_dist)"),
    ]
    pd.DataFrame(results).to_csv(PROJECT_ROOT / "models" / "gd2heme_reranking_comparison.csv", index=False)
    print("\nSaved to models/gd2heme_reranking_comparison.csv")


if __name__ == "__main__":
    main()
