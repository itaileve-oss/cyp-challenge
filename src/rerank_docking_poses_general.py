"""Generalized version of rerank_docking_poses.py for the 3 non-CYP3A4
isoforms: re-ranks each ligand's 50-pose GalaxyDock2-HEME bank by fit to
Fe-coordination geometry (coord_first: lowest-energy pose among those
reaching <3.5A Fe distance, falling back to rank 1 if none do) -- the
strategy already validated as a real, robust improvement over trusting
rank 1 alone for CYP3A4 (6/8 repeats, lower variance).

Parsing logic (parse_fe_coord/parse_bank_energies/parse_bank_mol2/
fe_dist_for_pose/process_ligand) is receptor-agnostic and copied verbatim
from rerank_docking_poses.py; only the receptor list, manifest, run-dir
root, and target column are isoform-specific.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
RUNS_ROOT = PROJECT_ROOT / "data" / "docking_results"
TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv"
HETEROATOM_ELEMENTS = {"N", "O", "S"}

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
    out = []
    for l in lines:
        try:
            out.append(float(l.split()[1]))
        except ValueError:
            # Fortran fixed-width field overflow (e.g. "********") -- a rare
            # GalaxyDock formatting quirk when a pose's energy is too large
            # for its column; drop that pose rather than crash the whole run.
            out.append(float("nan"))
    return out


def parse_bank_mol2(mol2_path: Path) -> list[list[tuple[str, np.ndarray]]]:
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

    fe_dists = np.array([fe_dist_for_pose(atoms, fe)[0] for atoms in poses])
    energies = np.array(energies)
    # Fortran field-overflow poses (NaN energy) can never legitimately win a
    # min-energy selection; +inf keeps array length/index alignment with
    # poses/fe_dists intact while making them unselectable.
    energies = np.where(np.isnan(energies), np.inf, energies)

    rank1_energy, rank1_fe = energies[0], fe_dists[0]

    coord_mask = fe_dists < 3.5
    if coord_mask.any():
        idx = np.argmin(np.where(coord_mask, energies, np.inf))
    else:
        idx = 0
    coord_energy, coord_fe = energies[idx], fe_dists[idx]

    return {
        "variant_id": run_dir.name,
        "rank1_energy": rank1_energy, "rank1_fe_dist": rank1_fe, "rank1_coord": rank1_fe < 3.5,
        "coord_energy": coord_energy, "coord_fe_dist": coord_fe, "coord_coord": coord_fe < 3.5,
        "n_poses": n,
    }


def parse_receptor(isoform: str, receptor: str, receptor_pdb: Path) -> pd.DataFrame:
    fe = parse_fe_coord(receptor_pdb)
    run_root = RUNS_ROOT / f"gd2heme_runs_{isoform.lower()}_{receptor}"
    run_dirs = sorted(run_root.iterdir())
    rows = []
    for i, rd in enumerate(run_dirs, 1):
        if not rd.is_dir():
            continue
        r = process_ligand(rd, fe)
        if r is not None:
            r["receptor"] = receptor
            rows.append(r)
        if i % 1000 == 0:
            print(f"  [{isoform}/{receptor}] {i}/{len(run_dirs)}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    isoform = sys.argv[1] if len(sys.argv) > 1 else "CYP1A2"
    receptors = ISOFORM_RECEPTORS[isoform]
    manifest_csv = PROJECT_ROOT / "data" / "structures" / f"{isoform.lower()}_variant_manifest.csv"
    target = f"{isoform}_pIC50_direct_inhibition"

    print(f"Re-ranking bank poses for {isoform} ({len(receptors)} receptors)...")
    dfs = []
    for receptor, pdb in receptors.items():
        df_r = parse_receptor(isoform, receptor, pdb)
        print(f"  [{receptor}] {len(df_r)} ligand variants done")
        dfs.append(df_r)
    raw = pd.concat(dfs, ignore_index=True)
    out_csv = PROJECT_ROOT / "models" / f"{isoform.lower()}_gd2heme_reranked_raw.csv"
    raw.to_csv(out_csv, index=False)
    print(f"\nTotal: {len(raw)} rows -- saved to {out_csv}")

    n_changed_coord = (raw.coord_energy != raw.rank1_energy).sum()
    print(f"coord_first picked a non-rank1 pose for {n_changed_coord}/{len(raw)} ({n_changed_coord/len(raw):.1%})")
    print(f"Fe-coordination rate -- rank1: {raw.rank1_coord.mean():.3f}  coord_first: {raw.coord_coord.mean():.3f}")

    manifest = pd.read_csv(manifest_csv)
    raw = raw.merge(manifest, on="variant_id", how="left").dropna(subset=["Molecule_Name"])
    train = pd.read_csv(TRAIN_CSV)[["Molecule_Name", target]].dropna()
    raw = raw.merge(train, on="Molecule_Name", how="inner")
    print(f"Rows with true pIC50: {len(raw)}")


if __name__ == "__main__":
    main()
