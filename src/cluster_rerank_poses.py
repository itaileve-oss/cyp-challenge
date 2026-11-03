"""Cluster-based pose selection: instead of trusting GalaxyDock2-HEME's single
best-energy pose (rank 1), cluster the full 50-pose bank by ligand-heavy-atom
RMSD (direct coordinate RMSD -- poses already share the receptor's fixed frame,
no superposition needed) using greedy leader clustering (2.0A cutoff, standard
convention), then take the best-energy member of the LARGEST cluster as the
"consensus" pose. The idea: a real binding mode gets independently rediscovered
by many samples in the bank; a spuriously low-energy pose is usually a lonely
singleton.

Also recomputes rank1 / coord_first / soft_penalty from rerank_docking_poses.py
in the same pass for a single, self-consistent comparison.
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
LAMBDA = 0.3
CLUSTER_CUTOFF = 2.0  # Angstrom, standard AutoDock-style leader clustering

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


def parse_bank_mol2_coords(mol2_path: Path):
    """Returns (elements, coords[n_pose, n_atom, 3]) -- heavy atoms, assumes
    identical atom count/order across poses (same ligand, different placement)."""
    text = mol2_path.read_text()
    blocks = text.split("@<TRIPOS>MOLECULE")[1:]
    elements = None
    all_coords = []
    for block in blocks:
        lines = block.splitlines()
        try:
            start = lines.index("@<TRIPOS>ATOM") + 1
        except ValueError:
            continue
        els, coords = [], []
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
        if elements is None:
            elements = els
        all_coords.append(coords)
    if not all_coords:
        return None, None
    n = min(len(c) for c in all_coords)
    coords_arr = np.array([c[:n] for c in all_coords])  # (n_pose, n_atom, 3)
    return elements[:n], coords_arr


def fe_dists_for_all_poses(elements, coords, fe):
    het_mask = np.array([e in HETEROATOM_ELEMENTS for e in elements])
    if not het_mask.any():
        d = np.linalg.norm(coords - fe, axis=-1).min(axis=1)
        return d
    het_coords = coords[:, het_mask, :]  # (n_pose, n_het, 3)
    d = np.linalg.norm(het_coords - fe, axis=-1).min(axis=1)
    return d


def leader_cluster(energies: np.ndarray, coords: np.ndarray, cutoff: float):
    """Greedy leader clustering on direct per-atom coordinate RMSD (no
    superposition -- poses already share the receptor's fixed frame)."""
    n_pose, n_atom, _ = coords.shape
    order = np.argsort(energies)
    diff = coords[:, None, :, :] - coords[None, :, :, :]
    rmsd = np.sqrt((diff ** 2).sum(-1).mean(-1))  # (n_pose, n_pose)

    cluster_id = -np.ones(n_pose, dtype=int)
    leaders = []
    for idx in order:
        assigned = -1
        for ci, leader_idx in enumerate(leaders):
            if rmsd[idx, leader_idx] < cutoff:
                assigned = ci
                break
        if assigned == -1:
            leaders.append(idx)
            assigned = len(leaders) - 1
        cluster_id[idx] = assigned
    return cluster_id, leaders


def process_ligand(run_dir: Path, fe: np.ndarray) -> dict | None:
    einfo = run_dir / "GD2_HEME_fb.E.info"
    mol2 = run_dir / "GD2_HEME_fb.mol2"
    if not einfo.exists() or not mol2.exists():
        return None
    energies = np.array(parse_bank_energies(einfo))
    elements, coords = parse_bank_mol2_coords(mol2)
    if elements is None:
        return None
    n = min(len(energies), coords.shape[0])
    energies, coords = energies[:n], coords[:n]
    fe_dists = fe_dists_for_all_poses(elements, coords, fe)

    rank1_energy, rank1_fe = energies[0], fe_dists[0]

    coord_mask = fe_dists < 3.5
    idx_coord = int(np.argmin(np.where(coord_mask, energies, np.inf))) if coord_mask.any() else 0
    coord_energy, coord_fe = energies[idx_coord], fe_dists[idx_coord]

    soft_score = energies + LAMBDA * fe_dists
    idx_soft = int(np.argmin(soft_score))
    soft_energy, soft_fe = energies[idx_soft], fe_dists[idx_soft]

    cluster_id, leaders = leader_cluster(energies, coords, CLUSTER_CUTOFF)
    counts = np.bincount(cluster_id, minlength=len(leaders))
    largest_ci = int(np.argmax(counts))
    idx_cluster = leaders[largest_ci]
    cluster_energy, cluster_fe = energies[idx_cluster], fe_dists[idx_cluster]
    cluster_pop = int(counts[largest_ci])
    rank1_in_largest = bool(cluster_id[0] == largest_ci)

    return {
        "variant_id": run_dir.name, "n_poses": n, "n_clusters": len(leaders),
        "rank1_energy": rank1_energy, "rank1_fe_dist": rank1_fe,
        "coord_energy": coord_energy, "coord_fe_dist": coord_fe,
        "soft_energy": soft_energy, "soft_fe_dist": soft_fe,
        "cluster_energy": cluster_energy, "cluster_fe_dist": cluster_fe,
        "cluster_pop": cluster_pop, "cluster_pop_frac": cluster_pop / n,
        "rank1_in_largest_cluster": rank1_in_largest,
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
    print("Cluster-based re-ranking for all 5 receptors...")
    dfs = []
    for receptor in RECEPTOR_PDBS:
        df_r = parse_receptor(receptor)
        print(f"  [{receptor}] {len(df_r)} ligand variants done, "
              f"mean n_clusters={df_r.n_clusters.mean():.1f}, "
              f"rank1-in-largest-cluster rate={df_r.rank1_in_largest_cluster.mean():.3f}")
        dfs.append(df_r)
    raw = pd.concat(dfs, ignore_index=True)
    raw.to_csv(PROJECT_ROOT / "models" / "gd2heme_clustered_raw.csv", index=False)
    print(f"\nTotal: {len(raw)} rows -- saved to models/gd2heme_clustered_raw.csv")
    print(f"\nOverall: mean poses/bank={raw.n_poses.mean():.1f}, mean clusters/bank={raw.n_clusters.mean():.1f}")
    print(f"Rank-1 already in the largest cluster: {raw.rank1_in_largest_cluster.mean():.1%} of the time")
    print(f"Mean largest-cluster population fraction: {raw.cluster_pop_frac.mean():.1%} of the 50-pose bank")

    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    train = pd.read_csv(TRAIN_CSV)[["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"]].dropna()
    raw = raw.merge(train, on="Molecule_Name", how="inner")
    TARGET = "CYP3A4_pIC50_direct_inhibition"
    print(f"\nRows with true pIC50: {len(raw)}")

    def eval_strategy(energy_col, fe_col, label):
        per_receptor = raw.groupby(["Molecule_Name", "receptor", TARGET], as_index=False).agg(
            energy=(energy_col, "mean"), fe=(fe_col, "mean"))
        agg = per_receptor.groupby(["Molecule_Name", TARGET], as_index=False).agg(
            energy=("energy", "mean"), fe=("fe", "mean"))
        agg["coord"] = agg.fe < 3.5
        r, p = pearsonr(agg.energy, agg[TARGET])
        rho, ps = spearmanr(agg.energy, agg[TARGET])
        print(f"[{label}] n={len(agg)}  Pearson={r:.4f} (p={p:.2e})  Spearman={rho:.4f}  Fe-coord frac={agg.coord.mean():.3f}")
        return {"label": label, "n": len(agg), "pearson": r, "pearson_p": p, "spearman": rho, "fe_coord_frac": agg.coord.mean()}

    print("\n=== Strategy comparison (mean-of-stereo per receptor, then mean-across-receptors) ===")
    results = [
        eval_strategy("rank1_energy", "rank1_fe_dist", "baseline (rank 1 / top GD2 energy)"),
        eval_strategy("coord_energy", "coord_fe_dist", "coord_first (hard 3.5A cutoff)"),
        eval_strategy("soft_energy", "soft_fe_dist", f"soft_penalty (energy + {LAMBDA}*fe_dist)"),
        eval_strategy("cluster_energy", "cluster_fe_dist", f"cluster_pop (largest cluster, {CLUSTER_CUTOFF}A cutoff)"),
    ]
    pd.DataFrame(results).to_csv(PROJECT_ROOT / "models" / "gd2heme_all_strategies_comparison.csv", index=False)
    print("\nSaved to models/gd2heme_all_strategies_comparison.csv")


if __name__ == "__main__":
    main()
