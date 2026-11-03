"""Extract the actual 3D coordinates of the coord_first-selected winning pose
(best receptor, best bank pose by the same hard-3.5A-cutoff-then-energy rule as
rerank_docking_poses.py) for every CYP3A4 training compound. Needed because the
earlier rerank pass only kept summary scalars (energy, fe_dist), not which pose.

Output: models/winning_docked_poses.pkl -- dict Molecule_Name -> {"elements":
[...], "coords": np.ndarray (n_atom,3), "receptor": str, "energy": float,
"fe_dist": float}
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
RUNS_ROOT = PROJECT_ROOT / "tmp" / "gd2heme_ensemble_20260920"
HETEROATOM_ELEMENTS = {"N", "O", "S"}


def parse_bank_energies(einfo_path: Path) -> list[float]:
    lines = [
        l for l in einfo_path.read_text().splitlines()
        if l.strip() and not l.startswith("!") and l.split()[0].isdigit()
    ]
    return [float(l.split()[1]) for l in lines]


def parse_bank_mol2(mol2_path: Path):
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
        raw_block = "@<TRIPOS>MOLECULE" + block
        poses.append((els, np.array(coords), raw_block))
    return poses


def fe_dist(elements, coords, fe):
    het_mask = np.array([e in HETEROATOM_ELEMENTS for e in elements])
    if not het_mask.any():
        return float(np.linalg.norm(coords - fe, axis=-1).min())
    return float(np.linalg.norm(coords[het_mask] - fe, axis=-1).min())


def parse_fe_coord(receptor_pdb: Path) -> np.ndarray:
    for line in receptor_pdb.read_text().splitlines():
        if line.startswith("HETATM") and line[76:78].strip() == "Fe":
            return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    raise ValueError(f"No Fe atom found in {receptor_pdb}")


RECEPTOR_PDBS = {
    "4ny4": PROJECT_ROOT / "data" / "receptors" / "cyp3a4_receptor_prepared.pdb",
    "1w0f": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_1w0f_prepared.pdb",
    "1w0g": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_1w0g_prepared.pdb",
    "3nxu": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_3nxu_prepared.pdb",
    "2j0d": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_2j0d_prepared.pdb",
}


def main():
    winners = pd.read_csv("/tmp/winning_poses.csv")
    fe_coords = {r: parse_fe_coord(p) for r, p in RECEPTOR_PDBS.items()}

    out = {}
    n_err = 0
    for i, row in enumerate(winners.itertuples(index=False), 1):
        run_dir = RUNS_ROOT / f"gd2heme_runs_{row.receptor}" / row.variant_id
        einfo = run_dir / "GD2_HEME_fb.E.info"
        mol2 = run_dir / "GD2_HEME_fb.mol2"
        try:
            energies = np.array(parse_bank_energies(einfo))
            poses = parse_bank_mol2(mol2)
            n = min(len(energies), len(poses))
            energies = energies[:n]
            fe = fe_coords[row.receptor]
            fe_dists = np.array([fe_dist(els, c, fe) for els, c, _ in poses[:n]])

            coord_mask = fe_dists < 3.5
            idx = int(np.argmin(np.where(coord_mask, energies, np.inf))) if coord_mask.any() else 0
            els, coords, raw_block = poses[idx]

            out[row.Molecule_Name] = {
                "elements": els, "coords": coords, "receptor": row.receptor,
                "energy": float(energies[idx]), "fe_dist": float(fe_dists[idx]),
                "mol2_block": raw_block,
            }
        except Exception as e:
            n_err += 1
            print(f"error on {row.Molecule_Name}: {e}")
        if i % 500 == 0:
            print(f"[{i}/{len(winners)}]", flush=True)

    print(f"\nExtracted {len(out)}/{len(winners)} poses ({n_err} errors)")
    with open(PROJECT_ROOT / "models" / "winning_docked_poses.pkl", "wb") as f:
        pickle.dump(out, f)
    print("Saved to models/winning_docked_poses.pkl")


if __name__ == "__main__":
    main()
