"""Pharmacophore features from Kaur/Sevrioukova/Poulos 2016 (J Med Chem 59:4210,
PMC4998966), operationalized on the real docked poses (not a free-ligand
conformer -- the specific mismatch that sank the earlier 2-point pharmacophore
attempt).

Their Figure 2 model: I = heme-ligating N donor (pyridine-type preferred);
II = flexible-backbone attachment, 8.5A from I; III (aromatic) and IV
(hydrophobic), each 5A from II at 110 degrees from each other; V (H-bond
donor/acceptor), 3A from II; VI (terminal group), 8-10A further out.

We can't recover the exact 3D angles from a 2D schematic figure with any
confidence, so this is a distance-based approximation, not a literal
reproduction: from the pose's own heme-coordinating atom (I, already
identified via the existing coord_fe_dist infrastructure), measure the
distance to the nearest OTHER aromatic ring centroid and to the nearest
amide-like H-bond donor/acceptor group centroid. Combining I-II (8.5A) with
II-III/V (5A/3A) via an unknown angle gives a plausible I-to-feature window
of roughly 4-13A (aromatic/hydrophobic) and 6-12A (H-bond group) -- wide
tolerance bands, not the paper's precise numbers.

Reuses the atom-index mapping from features_docked_electrostatics.py (pose
mol2 atom order -> training-SMILES RDKit atom order) so 2D-derived ring/amide
annotations (reliable) can be transferred onto the pose's 3D coordinates
(where re-deriving aromaticity directly is unreliable -- OpenBabel/RDKit
kekulization disagreements, same issue hit and only partly fixed there).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from features_docked_electrostatics import _prepare_docked_xyz, _topology_only  # noqa: E402

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
WINNING_POSES_PKL = PROJECT_ROOT / "models" / "winning_docked_poses.pkl"
HETEROATOM_ELEMENTS = {"N", "O", "S"}
AMIDE_SMARTS = Chem.MolFromSmarts("[CX3](=O)[NX3]")

FEATURE_COLS = [
    "nearest_aromatic_dist", "nearest_amide_dist", "is_coordinating",
    "nearest_hydrophobic_dist", "terminal_dist",
]


def _pose_mol_with_coords(pose_entry: dict):
    """Rebuild the pose as an RDKit Mol (own atom order, own coordinates),
    reusing the same radical/valence fix as features_docked_electrostatics.py."""
    pose_mol = Chem.MolFromMol2Block(pose_entry["mol2_block"], sanitize=False, removeHs=False)
    if pose_mol is None:
        return None
    for atom in pose_mol.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumRadicalElectrons(0)
    try:
        pose_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(pose_mol)
        pose_mol = Chem.RemoveHs(pose_mol)
    except Exception:
        return None
    return pose_mol


def _empty_result():
    return (np.nan, np.nan, 0.0, np.nan, np.nan)


def _hydrophobic_fragments(orig_mol: Chem.Mol) -> list[list[int]]:
    """Connected components of non-aromatic carbons with no directly-bonded
    heteroatom (simple lipophilic criterion -- e.g. cyclohexyl, isopropyl,
    tert-butyl, alkyl chains), size >=3, distinct from the aromatic rings
    used for feature III."""
    hydrophobic_idx = set()
    for atom in orig_mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetIsAromatic():
            continue
        if any(nbr.GetSymbol() not in ("C", "H") for nbr in atom.GetNeighbors()):
            continue
        hydrophobic_idx.add(atom.GetIdx())

    seen = set()
    fragments = []
    for start in hydrophobic_idx:
        if start in seen:
            continue
        stack, frag = [start], []
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            frag.append(i)
            for nbr in orig_mol.GetAtomWithIdx(i).GetNeighbors():
                if nbr.GetIdx() in hydrophobic_idx and nbr.GetIdx() not in seen:
                    stack.append(nbr.GetIdx())
        if len(frag) >= 3:
            fragments.append(frag)
    return fragments


def _one_compound(name: str, smiles: str, pose_entry: dict | None, fe: np.ndarray) -> tuple:
    if pose_entry is None:
        return _empty_result()

    orig_mol = Chem.MolFromSmiles(smiles)
    if orig_mol is None:
        return _empty_result()

    pose_mol = _pose_mol_with_coords(pose_entry)
    if pose_mol is None or pose_mol.GetNumAtoms() != orig_mol.GetNumAtoms():
        return _empty_result()

    match = orig_mol.GetSubstructMatch(pose_mol)
    if len(match) != orig_mol.GetNumAtoms():
        match = orig_mol.GetSubstructMatch(pose_mol, useChirality=False)
    if len(match) != orig_mol.GetNumAtoms():
        try:
            match = _topology_only(orig_mol).GetSubstructMatch(_topology_only(pose_mol))
        except Exception:
            match = ()
    if len(match) != orig_mol.GetNumAtoms():
        return _empty_result()
    # match[i] = orig-mol atom index for the i-th pose atom
    pose_to_orig = {i: match[i] for i in range(len(match))}

    coords = pose_entry["coords"]
    elements = pose_entry["elements"]
    n = min(len(elements), pose_mol.GetNumAtoms(), coords.shape[0])

    # identify the coordinating atom directly in pose coordinates (same rule as
    # the rest of the project: nearest N/O/S to Fe, <3.5A)
    het_idx = [i for i in range(n) if elements[i] in HETEROATOM_ELEMENTS]
    if not het_idx:
        return _empty_result()
    dists_to_fe = [np.linalg.norm(coords[i] - fe) for i in het_idx]
    coord_i = het_idx[int(np.argmin(dists_to_fe))]
    if dists_to_fe[int(np.argmin(dists_to_fe))] >= 3.5:
        return _empty_result()
    coord_xyz = coords[coord_i]

    # aromatic ring membership + amide matches, from the reliable 2D orig_mol,
    # transferred onto pose atom indices via pose_to_orig
    ri = orig_mol.GetRingInfo()
    aromatic_rings = [
        ring for ring in ri.AtomRings()
        if all(orig_mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
    ]
    amide_matches = orig_mol.GetSubstructMatches(AMIDE_SMARTS)

    orig_to_pose = {v: k for k, v in pose_to_orig.items()}

    def orig_ring_to_pose_centroid(ring_orig_idx):
        pose_idx = [orig_to_pose[a] for a in ring_orig_idx if a in orig_to_pose and orig_to_pose[a] < n]
        if not pose_idx:
            return None
        return coords[pose_idx].mean(axis=0)

    # nearest OTHER aromatic ring centroid (excluding the ring the coordinating atom itself sits in)
    coord_orig_idx = pose_to_orig.get(coord_i)
    best_arom_dist = np.nan
    for ring in aromatic_rings:
        if coord_orig_idx is not None and coord_orig_idx in ring:
            continue
        centroid = orig_ring_to_pose_centroid(ring)
        if centroid is None:
            continue
        d = float(np.linalg.norm(centroid - coord_xyz))
        if np.isnan(best_arom_dist) or d < best_arom_dist:
            best_arom_dist = d

    best_amide_dist = np.nan
    for match_atoms in amide_matches:
        centroid = orig_ring_to_pose_centroid(match_atoms)
        if centroid is None:
            continue
        d = float(np.linalg.norm(centroid - coord_xyz))
        if np.isnan(best_amide_dist) or d < best_amide_dist:
            best_amide_dist = d

    # IV: nearest non-aromatic hydrophobic fragment (>=3 contiguous aliphatic
    # carbons), distinct from III's aromatic rings
    best_hydrophobic_dist = np.nan
    for frag in _hydrophobic_fragments(orig_mol):
        centroid = orig_ring_to_pose_centroid(frag)
        if centroid is None:
            continue
        d = float(np.linalg.norm(centroid - coord_xyz))
        if np.isnan(best_hydrophobic_dist) or d < best_hydrophobic_dist:
            best_hydrophobic_dist = d

    # VI: terminal group -- the molecule's topologically-furthest heavy atom
    # from the coordinating atom, measured as real 3D distance in the docked pose
    terminal_dist = np.nan
    if coord_orig_idx is not None:
        dmat = Chem.GetDistanceMatrix(orig_mol)
        terminal_orig_idx = int(np.argmax(dmat[coord_orig_idx]))
        if terminal_orig_idx in orig_to_pose and orig_to_pose[terminal_orig_idx] < n:
            terminal_xyz = coords[orig_to_pose[terminal_orig_idx]]
            terminal_dist = float(np.linalg.norm(terminal_xyz - coord_xyz))

    return (best_arom_dist, best_amide_dist, 1.0, best_hydrophobic_dist, terminal_dist)


def pharmacophore_feature_matrix(molecule_names: list[str], smiles_list: list[str]) -> np.ndarray:
    with open(WINNING_POSES_PKL, "rb") as f:
        poses = pickle.load(f)

    RECEPTOR_PDBS = {
        "4ny4": PROJECT_ROOT / "data" / "receptors" / "cyp3a4_receptor_prepared.pdb",
        "1w0f": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_1w0f_prepared.pdb",
        "1w0g": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_1w0g_prepared.pdb",
        "3nxu": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_3nxu_prepared.pdb",
        "2j0d": PROJECT_ROOT / "data" / "receptors" / "ensemble" / "cyp3a4_2j0d_prepared.pdb",
    }

    def parse_fe(p):
        for line in Path(p).read_text().splitlines():
            if line.startswith("HETATM") and line[76:78].strip() == "Fe":
                return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    fe_coords = {r: parse_fe(p) for r, p in RECEPTOR_PDBS.items()}

    rows = []
    n_ok = 0
    for name, smiles in zip(molecule_names, smiles_list):
        entry = poses.get(name)
        fe = fe_coords[entry["receptor"]] if entry is not None else None
        r = _one_compound(name, smiles, entry, fe) if entry is not None else _empty_result()
        if not np.isnan(r[0]) or not np.isnan(r[1]):
            n_ok += 1
        rows.append(r)
    print(f"  [features_pharmacophore_kaur] usable geometry for {n_ok}/{len(molecule_names)} compounds")

    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    dist_cols = ["nearest_aromatic_dist", "nearest_amide_dist", "nearest_hydrophobic_dist", "terminal_dist"]
    means = df[dist_cols].mean()
    for col in dist_cols:
        df[col] = df[col].fillna(means[col])
    return df[FEATURE_COLS].to_numpy(dtype=float)
