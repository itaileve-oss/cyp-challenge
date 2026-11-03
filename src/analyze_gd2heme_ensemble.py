"""Parse the 5-receptor ensemble x stereo-enumerated GalaxyDock2-HEME docking
results, aggregate across receptors/stereoisomers, and correlate against true
CYP3A4 pIC50. Follow-up to analyze_gd2heme_results.py (single 4ny4 receptor,
one arbitrary stereoisomer per ligand -- only 3% Fe-coordination, r=-0.284).

Tests two open questions from that run's diagnosis:
  1. Does docking against an ENSEMBLE of 5 receptor conformations (4ny4, 1w0f,
     1w0g, 3nxu, 2j0d) recover more real heme-Fe-coordinating poses than a
     single receptor did?
  2. Best-of vs mean-of aggregation, both across receptors and across the
     enumerated-stereoisomer variants of each compound.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
RUNS_ROOT = PROJECT_ROOT / "tmp" / "gd2heme_ensemble_20260920"
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "structures" / "variant_manifest.csv"
BOX_CENTER = np.array([-22.179, -17.983, -9.575])
HETEROATOM_ELEMENTS = {"N", "O", "S"}

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
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            return np.array([x, y, z])
    raise ValueError(f"No Fe atom found in {receptor_pdb}")


def parse_mol2_atoms(mol2_path: Path) -> list[tuple[str, np.ndarray]]:
    lines = mol2_path.read_text().splitlines()
    start = lines.index("@<TRIPOS>ATOM") + 1
    atoms = []
    for line in lines[start:]:
        if line.startswith("@<TRIPOS>"):
            break
        parts = line.split()
        sybyl = parts[5]
        element = sybyl.split(".")[0]
        if element == "H" or sybyl.startswith("H"):
            element = "H"
        xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        atoms.append((element, xyz))
    return atoms


def parse_best_energy(einfo_path: Path) -> dict:
    text = einfo_path.read_text()
    data_lines = [
        l for l in text.splitlines()
        if l.strip() and not l.startswith("!") and l.split()[0].isdigit()
    ]
    best = data_lines[0].split()
    names = ["bank_no", "energy", "l_rmsd", "atdk_e", "int_e", "ds_e", "hm_e", "plp", "nc_e", "nr_e", "sc_e", "oc_e"]
    vals = [float(x) for x in best]
    return dict(zip(names, vals))


def analyze_one(run_dir: Path, fe: np.ndarray) -> dict | None:
    einfo = run_dir / "GD2_HEME_fb.E.info"
    mol2 = run_dir / "GD2_HEME_fb.mol2"
    if not einfo.exists() or not mol2.exists():
        return None

    scores = parse_best_energy(einfo)
    atoms = parse_mol2_atoms(mol2)
    heavy = [(e, xyz) for e, xyz in atoms if e != "H"]
    centroid = np.mean([xyz for _, xyz in heavy], axis=0)
    dist_to_box_center = float(np.linalg.norm(centroid - BOX_CENTER))

    het_dists = [
        (e, float(np.linalg.norm(xyz - fe)))
        for e, xyz in heavy
        if e in HETEROATOM_ELEMENTS
    ]
    if het_dists:
        closest_elem, closest_dist = min(het_dists, key=lambda t: t[1])
    else:
        closest_elem, closest_dist = None, float(min(np.linalg.norm(xyz - fe) for _, xyz in heavy))

    return {
        "variant_id": run_dir.name,
        "docking_energy": scores["energy"],
        "hm_e": scores["hm_e"],
        "atdk_e": scores["atdk_e"],
        "int_e": scores["int_e"],
        "ds_e": scores["ds_e"],
        "n_heavy_atoms": len(heavy),
        "centroid_dist_to_box_center": dist_to_box_center,
        "fe_closest_het_elem": closest_elem,
        "fe_closest_het_dist": closest_dist,
        "fe_coordinating": closest_dist < 3.5,
    }


def parse_receptor(receptor: str) -> pd.DataFrame:
    fe = parse_fe_coord(RECEPTOR_PDBS[receptor])
    run_dirs = sorted((RUNS_ROOT / f"gd2heme_runs_{receptor}").iterdir())
    rows = []
    n_err = 0
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        try:
            r = analyze_one(rd, fe)
        except Exception as e:
            n_err += 1
            r = None
        if r is not None:
            r["receptor"] = receptor
            rows.append(r)
    if n_err:
        print(f"  [{receptor}] {n_err} parse errors")
    return pd.DataFrame(rows)


def main():
    print("Parsing all 5 receptors...")
    dfs = []
    for receptor in RECEPTOR_PDBS:
        df_r = parse_receptor(receptor)
        print(f"  [{receptor}] {len(df_r)} poses parsed")
        dfs.append(df_r)
    raw = pd.concat(dfs, ignore_index=True)
    raw.to_csv(PROJECT_ROOT / "models" / "gd2heme_ensemble_raw.csv", index=False)
    print(f"\nTotal raw poses: {len(raw)} -- saved to models/gd2heme_ensemble_raw.csv")

    print("\n--- Heme-Fe coordination, ALL poses (vs real-structure benchmark: median 2.44A, 83% <3.5A) ---")
    print(f"Median closest-heteroatom-to-Fe distance: {raw.fe_closest_het_dist.median():.2f} A")
    print(f"Fraction coordinating (<3.5A), all poses: {raw.fe_coordinating.mean():.3f}")
    print("Per-receptor breakdown:")
    print(raw.groupby("receptor").fe_coordinating.mean())

    manifest = pd.read_csv(MANIFEST_CSV)
    raw = raw.merge(manifest, on="variant_id", how="left")
    n_unmapped = raw.Molecule_Name.isna().sum()
    if n_unmapped:
        print(f"WARNING: {n_unmapped} poses have no Molecule_Name in manifest")
    raw = raw.dropna(subset=["Molecule_Name"])

    train = pd.read_csv(TRAIN_CSV)[["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"]].dropna()
    raw = raw.merge(train, on="Molecule_Name", how="inner")
    print(f"\nPoses with a true CYP3A4 pIC50 label: {len(raw)}")

    def eval_strategy(df_agg: pd.DataFrame, label: str) -> dict:
        r, p = pearsonr(df_agg.docking_energy, df_agg.CYP3A4_pIC50_direct_inhibition)
        rho, ps = spearmanr(df_agg.docking_energy, df_agg.CYP3A4_pIC50_direct_inhibition)
        r_hm, p_hm = pearsonr(df_agg.hm_e, df_agg.CYP3A4_pIC50_direct_inhibition)
        coord_frac = df_agg.fe_coordinating.mean() if "fe_coordinating" in df_agg else float("nan")
        print(f"[{label}] n={len(df_agg)}  Pearson(energy)={r:.3f} (p={p:.2e})  "
              f"Spearman={rho:.3f} (p={ps:.2e})  Pearson(hm_e)={r_hm:.3f}  "
              f"Fe-coord frac={coord_frac:.3f}")
        return {"label": label, "n": len(df_agg), "pearson_energy": r, "pearson_p": p,
                "spearman": rho, "spearman_p": ps, "pearson_hm_e": r_hm, "fe_coord_frac": coord_frac}

    print("\n=== Aggregation strategy comparison (docking_energy vs true pIC50) ===")
    results = []

    # Strategy A: best pose overall (min energy across ALL receptor x stereo combos)
    best_overall = raw.loc[raw.groupby("Molecule_Name").docking_energy.idxmin()]
    results.append(eval_strategy(best_overall, "best-of-all (min energy, any receptor+stereo)"))

    # Strategy B: mean energy across ALL receptor x stereo combos
    mean_overall = raw.groupby(["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"], as_index=False).agg(
        docking_energy=("docking_energy", "mean"), hm_e=("hm_e", "mean"),
        fe_coordinating=("fe_coordinating", "mean"),
    )
    results.append(eval_strategy(mean_overall, "mean-of-all (mean energy, all receptor+stereo)"))

    # Strategy C: best stereo variant per receptor, then MEAN across receptors (ensemble-average)
    best_per_receptor = raw.loc[raw.groupby(["Molecule_Name", "receptor"]).docking_energy.idxmin()]
    ensemble_mean = best_per_receptor.groupby(["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"], as_index=False).agg(
        docking_energy=("docking_energy", "mean"), hm_e=("hm_e", "mean"),
        fe_coordinating=("fe_coordinating", "mean"),
    )
    results.append(eval_strategy(ensemble_mean, "best-stereo-per-receptor, then mean-across-receptors"))

    # Strategy D: best stereo variant per receptor, then MIN across receptors (best-of-ensemble)
    ensemble_min = best_per_receptor.loc[best_per_receptor.groupby("Molecule_Name").docking_energy.idxmin()]
    results.append(eval_strategy(ensemble_min, "best-stereo-per-receptor, then min-across-receptors"))

    # Strategy E: mean stereo variant per receptor, then mean across receptors
    mean_per_receptor = raw.groupby(["Molecule_Name", "receptor", "CYP3A4_pIC50_direct_inhibition"], as_index=False).agg(
        docking_energy=("docking_energy", "mean"), hm_e=("hm_e", "mean"),
        fe_coordinating=("fe_coordinating", "mean"),
    )
    ensemble_mean_mean = mean_per_receptor.groupby(["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"], as_index=False).agg(
        docking_energy=("docking_energy", "mean"), hm_e=("hm_e", "mean"),
        fe_coordinating=("fe_coordinating", "mean"),
    )
    results.append(eval_strategy(ensemble_mean_mean, "mean-stereo-per-receptor, then mean-across-receptors"))

    summary = pd.DataFrame(results)
    summary.to_csv(PROJECT_ROOT / "models" / "gd2heme_ensemble_strategy_comparison.csv", index=False)
    print("\nSaved strategy comparison to models/gd2heme_ensemble_strategy_comparison.csv")

    best_overall.to_csv(PROJECT_ROOT / "models" / "gd2heme_ensemble_best_overall_features.csv", index=False)
    print("Saved best-of-all per-compound features to models/gd2heme_ensemble_best_overall_features.csv")


if __name__ == "__main__":
    main()
