"""Docked-pose CYP1A2 pharmacophore: planarity (RDKit's plane-of-best-fit
deviation, CalcPBF -- lower = flatter) measured on the REAL docked pose
instead of a free MMFF conformer. CYP1A2's active site is small, narrow and
flat, so how flat a molecule sits when actually bound should be a more
faithful signal than how flat its unbound lowest-energy conformer happens to
be -- the same upgrade that mattered for CYP3A4's and CYP2D6's pharmacophores.
Also adds the docked pose's centroid distance to the heme iron (how close/deep
the (flat) molecule sits near the oxidation site), a second geometry feature
in the same spirit as CYP2D6's winning distance-to-Fe measurement.

Unlike the CYP2C9/CYP2D6 docked features, this one needs no atom-mapping back
to the original SMILES -- PBF and centroid distance are whole-molecule
geometry properties of the pose's own coordinates, regardless of atom order.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

FEATURE_COLS = ["docked_pbf", "docked_centroid_to_fe_dist"]


def _empty_result():
    return (np.nan, np.nan)


def _one_compound(pose_entry: dict | None, fe: np.ndarray) -> tuple:
    if pose_entry is None:
        return _empty_result()

    pose_mol = Chem.MolFromMol2Block(pose_entry["mol2_block"], sanitize=False, removeHs=True)
    if pose_mol is None:
        return _empty_result()
    try:
        pose_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(pose_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception:
        pass

    if pose_mol.GetNumConformers() == 0 or pose_mol.GetNumAtoms() < 4:
        return _empty_result()

    try:
        pbf = float(rdMolDescriptors.CalcPBF(pose_mol))
    except Exception:
        pbf = np.nan

    coords = pose_mol.GetConformer().GetPositions()
    centroid = coords.mean(axis=0)
    centroid_dist = float(np.linalg.norm(centroid - fe))

    return (pbf, centroid_dist)


def cyp1a2_docked_feature_matrix(molecule_names: list[str], poses: dict, fe_coords: dict) -> np.ndarray:
    rows = []
    n_ok = 0
    for name in molecule_names:
        entry = poses.get(name)
        fe = fe_coords[entry["receptor"]] if entry is not None else None
        r = _one_compound(entry, fe) if entry is not None else _empty_result()
        if not np.isnan(r[0]):
            n_ok += 1
        rows.append(r)
    print(f"  [features_pharmacophore_cyp1a2_docked] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
