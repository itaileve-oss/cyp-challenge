"""Re-extract HM_E (GalaxyDock2-HEME's heme-specific energy term) for the
coord_first-selected pose, per (compound, receptor). Not saved by the
original rerank_docking_poses.py pass (only scalar energy/fe_dist were kept),
so this re-scans the same 50-pose banks and re-applies the identical
selection rule (fe_dist<3.5 then min energy, else rank1) to recover which
bank index was chosen, then pulls its HM_E value.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
RUNS_ROOT = PROJECT_ROOT / "tmp" / "gd2heme_ensemble_20260920"
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
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    raise ValueError(f"No Fe atom found in {receptor_pdb}")


def parse_bank_full(einfo_path: Path):
    lines = [
        l for l in einfo_path.read_text().splitlines()
        if l.strip() and not l.startswith("!") and l.split()[0].isdigit()
    ]
    energies = np.array([float(l.split()[1]) for l in lines])
    hm_e = np.array([float(l.split()[6]) for l in lines])
    return energies, hm_e


def parse_bank_mol2_coords(mol2_path: Path):
    text = mol2_path.read_text()
    blocks = text.split("@<TRIPOS>MOLECULE")[1:]
    poses = []
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
        poses.append((els, np.array(coords)))
    return poses


def fe_dist_for_pose(els, coords, fe):
    het_mask = np.array([e in HETEROATOM_ELEMENTS for e in els])
    if not het_mask.any():
        return float(np.linalg.norm(coords - fe, axis=-1).min())
    return float(np.linalg.norm(coords[het_mask] - fe, axis=-1).min())


def process_ligand(run_dir: Path, fe: np.ndarray):
    einfo = run_dir / "GD2_HEME_fb.E.info"
    mol2 = run_dir / "GD2_HEME_fb.mol2"
    if not einfo.exists() or not mol2.exists():
        return None
    energies, hm_e_arr = parse_bank_full(einfo)
    poses = parse_bank_mol2_coords(mol2)
    n = min(len(energies), len(poses))
    energies, hm_e_arr = energies[:n], hm_e_arr[:n]
    fe_dists = np.array([fe_dist_for_pose(els, c, fe) for els, c in poses[:n]])

    coord_mask = fe_dists < 3.5
    idx = int(np.argmin(np.where(coord_mask, energies, np.inf))) if coord_mask.any() else 0
    return {
        "coord_energy": float(energies[idx]), "coord_fe_dist": float(fe_dists[idx]),
        "coord_hm_e": float(hm_e_arr[idx]),
    }


def main():
    print("Re-extracting HM_E for coord_first poses, all 5 receptors...", flush=True)
    rows = []
    for receptor, pdb_path in RECEPTOR_PDBS.items():
        fe = parse_fe_coord(pdb_path)
        run_dirs = sorted((RUNS_ROOT / f"gd2heme_runs_{receptor}").iterdir())
        for i, rd in enumerate(run_dirs, 1):
            if not rd.is_dir():
                continue
            r = process_ligand(rd, fe)
            if r is not None:
                r["receptor"] = receptor
                r["variant_id"] = rd.name
                rows.append(r)
            if i % 1000 == 0:
                print(f"  [{receptor}] {i}/{len(run_dirs)}", flush=True)
        print(f"  [{receptor}] done", flush=True)

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "models" / "gd2heme_hm_e_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
