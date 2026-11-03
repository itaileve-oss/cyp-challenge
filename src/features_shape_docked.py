"""Whole-molecule 3D shape descriptors (RDKit's standard battery: normalized
principal moment ratios NPR1/NPR2 -- classify along the rod/disc/sphere
triangle -- radius of gyration, asphericity, eccentricity, inertial shape
factor, spherocity index) computed on the REAL DOCKED POSE rather than a free
conformer. Physics-based, not tried anywhere in this project before.

No atom-mapping to the original SMILES needed -- these are whole-molecule
geometry properties of the pose's own coordinates (same reasoning as
features_pharmacophore_cyp1a2_docked.py's PBF).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

FEATURE_COLS = ["npr1", "npr2", "radius_of_gyration", "asphericity",
                "eccentricity", "inertial_shape_factor", "spherocity_index"]


def _empty_result():
    return tuple(np.nan for _ in FEATURE_COLS)


def _one_compound(pose_entry: dict | None) -> tuple:
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
        return (
            float(rdMolDescriptors.CalcNPR1(pose_mol)),
            float(rdMolDescriptors.CalcNPR2(pose_mol)),
            float(rdMolDescriptors.CalcRadiusOfGyration(pose_mol)),
            float(rdMolDescriptors.CalcAsphericity(pose_mol)),
            float(rdMolDescriptors.CalcEccentricity(pose_mol)),
            float(rdMolDescriptors.CalcInertialShapeFactor(pose_mol)),
            float(rdMolDescriptors.CalcSpherocityIndex(pose_mol)),
        )
    except Exception:
        return _empty_result()


def shape_docked_feature_matrix(molecule_names: list[str], poses: dict) -> np.ndarray:
    rows = []
    n_ok = 0
    for name in molecule_names:
        entry = poses.get(name)
        r = _one_compound(entry)
        if not np.isnan(r[0]):
            n_ok += 1
        rows.append(r)
    print(f"  [features_shape_docked] usable geometry for {n_ok}/{len(molecule_names)} compounds")
    return np.array(rows, dtype=float)
