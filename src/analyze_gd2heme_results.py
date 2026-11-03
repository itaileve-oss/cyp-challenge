"""Parse GalaxyDock2-HEME batch results, sanity-check poses, and correlate
docking score against true CYP3A4 pIC50.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_DIR = Path("/home/ileven/Projects/cyp-challenge/tmp/gd2heme_results_20260918/gd2heme_runs")
TRAIN_CSV = Path("/home/ileven/Projects/cyp-challenge/data/raw/TRAIN_inhibition.csv")
BOX_CENTER = np.array([-22.179, -17.983, -9.575])

HETEROATOM_ELEMENTS = {"N", "O", "S"}


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
    # header: Bank No Energy l_RMSD ATDK_E INT_E DS_E HM_E PLP NC_E NR_E SC_E OC_E
    names = ["bank_no", "energy", "l_rmsd", "atdk_e", "int_e", "ds_e", "hm_e", "plp", "nc_e", "nr_e", "sc_e", "oc_e"]
    vals = [float(x) for x in best]
    return dict(zip(names, vals))


def analyze_ligand(run_dir: Path) -> dict | None:
    einfo = run_dir / "GD2_HEME_fb.E.info"
    mol2 = run_dir / "GD2_HEME_fb.mol2"
    receptor = run_dir / "receptor.pdb"
    if not einfo.exists() or not mol2.exists():
        return None

    scores = parse_best_energy(einfo)
    fe = parse_fe_coord(receptor)
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
        "Molecule_Name": run_dir.name,
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


def main():
    run_dirs = sorted(RUNS_DIR.iterdir())
    rows = []
    for rd in run_dirs:
        if not rd.is_dir():
            continue
        try:
            r = analyze_ligand(rd)
        except Exception as e:
            print(f"parse error {rd.name}: {e}")
            r = None
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} ligand results")

    print("\n--- Pose sanity check ---")
    print(f"Centroid distance to box center: mean={df.centroid_dist_to_box_center.mean():.2f}, "
          f"median={df.centroid_dist_to_box_center.median():.2f}, "
          f"max={df.centroid_dist_to_box_center.max():.2f}")
    print(f"Fraction within 10A of box center: {(df.centroid_dist_to_box_center < 10).mean():.3f}")

    print("\n--- Heme-Fe coordination check (vs real-structure benchmark: median 2.44A, 83% <3.5A, N in 84%) ---")
    print(f"Median closest-heteroatom-to-Fe distance: {df.fe_closest_het_dist.median():.2f} A")
    print(f"Fraction coordinating (<3.5A): {df.fe_coordinating.mean():.3f}")
    coord_df = df[df.fe_coordinating]
    if len(coord_df):
        elem_counts = coord_df.fe_closest_het_elem.value_counts(normalize=True)
        print(f"Coordinating-atom element breakdown:\n{elem_counts}")

    df.to_csv("/home/ileven/Projects/cyp-challenge/models/gd2heme_docking_features.csv", index=False)
    print("\nSaved features to models/gd2heme_docking_features.csv")

    train = pd.read_csv(TRAIN_CSV)
    merged = df.merge(
        train[["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"]],
        on="Molecule_Name", how="inner",
    ).dropna(subset=["CYP3A4_pIC50_direct_inhibition"])
    print(f"\n--- Merged with true pIC50: {len(merged)} ligands ---")

    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(merged.docking_energy, merged.CYP3A4_pIC50_direct_inhibition)
    rho, ps = spearmanr(merged.docking_energy, merged.CYP3A4_pIC50_direct_inhibition)
    print(f"docking_energy vs pIC50: Pearson r={r:.3f} (p={p:.2e}), Spearman rho={rho:.3f} (p={ps:.2e})")

    r_hm, p_hm = pearsonr(merged.hm_e, merged.CYP3A4_pIC50_direct_inhibition)
    print(f"hm_e (heme term) vs pIC50: Pearson r={r_hm:.3f} (p={p_hm:.2e})")

    merged.to_csv("/home/ileven/Projects/cyp-challenge/models/gd2heme_merged_with_pic50.csv", index=False)
    print("Saved merged table to models/gd2heme_merged_with_pic50.csv")


if __name__ == "__main__":
    main()
