"""Structural contact features from the real Vina-GPU-docked poses
(tmp/docked_all/*.pdbqt), using the CYP3A4 hotspot geometry already identified
empirically from the 100 real bound crystal structures (analyze_query_poses.py):
heme-Fe coordination, SER119 H-bond, ARG212 salt bridge (95% of real ligands
contact it), and the Phe aromatic cage (57/108/213/215/241/304).

Why this is different from (and complementary to) features_pharmacophore.py:
that feature measured an INTRAmolecular distance on a free-ligand MMFF
conformer -- a proxy for "does this molecule have the right shape," but with
no guarantee the free-ligand low-energy conformer resembles the true
CYP3A4-bound conformation. Here we use the actual docked pose sitting inside
the real prepared CYP3A4 receptor, so these are literal protein-ligand contact
distances, not intramolecular proxies -- directly testing whether that
free-ligand-vs-bound-conformer mismatch was a real limitation.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKED_DIR = PROJECT_ROOT / "tmp" / "docked_all"
RECEPTOR_PDB = PROJECT_ROOT / "data" / "receptors" / "cyp3a4_receptor_prepared.pdb"

PHE_CLUSTER_RESIDUES = [57, 108, 213, 215, 241, 304]
PHE_RING_ATOMS = {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}


def _parse_receptor_landmarks(pdb_path: Path) -> dict:
    """Returns {'fe': (3,), 'ser119_og': (3,), 'arg212_guanidinium': (N,3),
    'phe_ring_centroids': (6,3)}."""
    fe = None
    ser119_og = None
    arg212_atoms = []
    phe_ring_atoms = {r: [] for r in PHE_CLUSTER_RESIDUES}

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21]
            try:
                res_num = int(line[22:26])
            except ValueError:
                continue
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            pos = np.array([x, y, z])

            if chain != "A":
                continue
            if res_name == "HEM" and atom_name == "FE":
                fe = pos
            elif res_name == "SER" and res_num == 119 and atom_name == "OG":
                ser119_og = pos
            elif res_name == "ARG" and res_num == 212 and atom_name in {"NE", "NH1", "NH2"}:
                arg212_atoms.append(pos)
            elif res_name == "PHE" and res_num in PHE_CLUSTER_RESIDUES and atom_name in PHE_RING_ATOMS:
                phe_ring_atoms[res_num].append(pos)

    phe_centroids = np.array([
        np.mean(phe_ring_atoms[r], axis=0) for r in PHE_CLUSTER_RESIDUES if phe_ring_atoms[r]
    ])

    return {
        "fe": fe,
        "ser119_og": ser119_og,
        "arg212": np.array(arg212_atoms) if arg212_atoms else None,
        "phe_centroids": phe_centroids,
    }


_ATOM_LINE_RE = re.compile(
    r"^(ATOM|HETATM)\s*\d+\s+\S+\s+\S+\s+\d+\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+).*?(\S+)\s*$"
)


def _parse_best_pose_atoms(pdbqt_path: Path) -> list:
    """Returns list of (element_or_autodock_type, np.array xyz) for MODEL 1 only."""
    atoms = []
    in_model_1 = False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                in_model_1 = line.strip() == "MODEL 1"
                continue
            if line.startswith("ENDMDL"):
                if in_model_1:
                    break
                continue
            if not in_model_1:
                continue
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atype = line[77:79].strip() if len(line) >= 79 else line.split()[-1]
                atoms.append((atype, np.array([x, y, z])))
    return atoms


# AutoDock atom types that carry a lone pair capable of heme-Fe coordination
# or H-bond acceptance (N/NA = nitrogen variants, OA/SA = O/S acceptor variants).
_COORD_TYPES = {"N", "NA", "OA", "SA"}
_HYDROPHOBIC_TYPES = {"C", "A"}  # aliphatic / aromatic carbon


def pose_contact_features(atoms: list, landmarks: dict) -> dict:
    result = {
        "dock_fe_min_dist": np.nan,
        "dock_ser119_min_dist": np.nan,
        "dock_arg212_min_dist": np.nan,
        "dock_phe_cage_contacts": np.nan,
    }
    if not atoms:
        return result

    coord_positions = np.array([pos for atype, pos in atoms if atype in _COORD_TYPES])
    hydrophobic_positions = np.array([pos for atype, pos in atoms if atype in _HYDROPHOBIC_TYPES])
    all_positions = np.array([pos for _, pos in atoms])

    if landmarks["fe"] is not None and len(coord_positions):
        result["dock_fe_min_dist"] = float(np.linalg.norm(coord_positions - landmarks["fe"], axis=1).min())

    if landmarks["ser119_og"] is not None and len(coord_positions):
        result["dock_ser119_min_dist"] = float(
            np.linalg.norm(coord_positions - landmarks["ser119_og"], axis=1).min()
        )

    if landmarks["arg212"] is not None and len(all_positions):
        dmat = np.linalg.norm(all_positions[:, None, :] - landmarks["arg212"][None, :, :], axis=2)
        result["dock_arg212_min_dist"] = float(dmat.min())

    if len(landmarks["phe_centroids"]) and len(hydrophobic_positions):
        dmat = np.linalg.norm(
            hydrophobic_positions[:, None, :] - landmarks["phe_centroids"][None, :, :], axis=2
        )
        result["dock_phe_cage_contacts"] = float((dmat < 4.5).any(axis=1).sum())

    return result


def build_docking_contact_table() -> pd.DataFrame:
    landmarks = _parse_receptor_landmarks(RECEPTOR_PDB)
    rows = []
    for path in sorted(DOCKED_DIR.glob("*_out.pdbqt")):
        mol_name = path.stem.removesuffix("_out")
        atoms = _parse_best_pose_atoms(path)
        feats = pose_contact_features(atoms, landmarks)
        feats["Molecule_Name"] = mol_name
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_docking_contact_table()
    out_path = PROJECT_ROOT / "models" / "cyp3a4_docking_contact_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.describe())
