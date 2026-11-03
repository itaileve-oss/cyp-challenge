"""Isoform-specific pharmacophore features for CYP2D6, CYP2C9, and CYP1A2,
analogous in spirit to the CYP3A4 heme/Ser119 two-point distance in
features_pharmacophore.py, but each built from that isoform's own well-established
active-site SAR rather than a shared geometric rule:

- CYP2D6: substrates/inhibitors almost universally carry a basic, protonatable
  nitrogen (piperidine/pyrrolidine/tertiary amine-type) that forms an ionic /
  H-bond interaction with Glu216 or Asp301 in the active site, typically 5-7 A
  from the heme iron (the oxidation site). This is one of the single best-known
  univariate CYP2D6 inhibition descriptors in the literature (e.g. de Groot et
  al.). We don't have CYP2D6-bound reference structures cached in this project
  (unlike the 100 aligned CYP3A4 poses used for the heme/Ser119 distance), so
  this starts as the coarser-but-robust "has a basic amine" flag plus, when a
  basic N is present, its distance to the nearest aromatic-ring centroid (a
  proxy for the planar region near the oxidation site that gets stacked in the
  active site) -- not a heme-anchored distance like CYP3A4's.

- CYP2C9: inhibitors commonly carry an anionic/acidic group (carboxylic acid,
  tetrazole, or acidic sulfonamide) that salt-bridges with Arg108 near the
  active-site entrance (classic examples: NSAIDs, warfarin). Encoded simply as
  presence/count of an acidic group -- no distance geometry, since the acidic
  group's role is at the entrance channel, not tied to a specific intramolecular
  distance the way CYP3A4's heme-coordination pharmacophore is.

- CYP1A2: the active site is small, narrow, and planar -- CYP1A2 substrates and
  inhibitors are characteristically flat, compact polyaromatic/heteroaromatic
  systems (caffeine, planar PAHs). Encoded as 3D planarity (RDKit's
  CalcPBF -- plane of best fit deviation, lower = flatter) computed on the
  lowest-MMFF-energy conformer, plus aromatic-atom fraction and ring count as
  simple 2D companions.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from features_pharmacophore import _lowest_energy_conf_id

# --- CYP2D6: basic amine ---------------------------------------------------
# Excludes amide/sulfonamide N (lone pair delocalized, not basic), aniline-type N
# (aromatic-ring-attached, basicity greatly reduced), and guanidine/amidine
# (different, much higher pKa chemistry, not the classic "basic amine" pharmacophore).
_BASIC_AMINE_N = Chem.MolFromSmarts(
    "[NX3;H0,H1,H2;+0;!$(NC=O);!$(NS(=O)=O);!$(N-c);!$(N=*);!$(N-C=[N,S])]"
)
_AROMATIC_RING = Chem.MolFromSmarts("[a]")

# --- CYP2C9: acidic group ---------------------------------------------------
_CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
_TETRAZOLE = Chem.MolFromSmarts("[nH]1nnnc1")
_SULFONIC_ACID = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[OX2H1]")
_ACIDIC_PATTERNS = [_CARBOXYLIC_ACID, _TETRAZOLE, _SULFONIC_ACID]


def cyp2d6_features(mol: Chem.Mol) -> dict:
    basic_n_idx = {m[0] for m in mol.GetSubstructMatches(_BASIC_AMINE_N)}
    has_basic_n = len(basic_n_idx) > 0
    result = {"has_basic_amine": has_basic_n, "basic_n_to_ring_dist": np.nan}
    if not has_basic_n or mol.GetNumConformers() == 0:
        return result

    ring_info = mol.GetRingInfo()
    aromatic_ring_atoms = [
        ring for ring in ring_info.AtomRings()
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
    ]
    if not aromatic_ring_atoms:
        return result

    cid = _lowest_energy_conf_id(mol)
    pos = mol.GetConformer(cid).GetPositions()
    basic_n_pos = pos[list(basic_n_idx)]  # (B, 3)
    centroids = np.array([pos[list(ring)].mean(axis=0) for ring in aromatic_ring_atoms])  # (R, 3)
    dmat = np.linalg.norm(basic_n_pos[:, None, :] - centroids[None, :, :], axis=2)
    result["basic_n_to_ring_dist"] = float(dmat.min())
    return result


def cyp2c9_features(mol: Chem.Mol) -> dict:
    n_acidic = 0
    for pat in _ACIDIC_PATTERNS:
        n_acidic += len(mol.GetSubstructMatches(pat))
    return {"has_acidic_group": n_acidic > 0, "n_acidic_groups": n_acidic}


def cyp1a2_features(mol: Chem.Mol) -> dict:
    n_atoms = mol.GetNumAtoms()
    aromatic_frac = (
        sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / n_atoms if n_atoms else 0.0
    )
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    result = {
        "aromatic_fraction": aromatic_frac,
        "n_aromatic_rings": n_aromatic_rings,
        "pbf": np.nan,
    }
    if mol.GetNumConformers() == 0:
        return result
    cid = _lowest_energy_conf_id(mol)
    try:
        result["pbf"] = float(rdMolDescriptors.CalcPBF(mol, confId=cid))
    except Exception:
        pass
    return result


def cyp2d6_feature_matrix(mols: list) -> "np.ndarray":
    """(N, 2): has_basic_amine, basic_n_to_ring_dist (sentinel 20.0 if absent)."""
    rows = []
    for mol in mols:
        if mol is None:
            rows.append([0.0, 20.0])
            continue
        f = cyp2d6_features(mol)
        dist = f["basic_n_to_ring_dist"]
        rows.append([float(f["has_basic_amine"]), 20.0 if np.isnan(dist) else dist])
    return np.array(rows, dtype=np.float32)


def cyp2c9_feature_matrix(mols: list) -> "np.ndarray":
    """(N, 2): has_acidic_group, n_acidic_groups."""
    rows = []
    for mol in mols:
        if mol is None:
            rows.append([0.0, 0.0])
            continue
        f = cyp2c9_features(mol)
        rows.append([float(f["has_acidic_group"]), float(f["n_acidic_groups"])])
    return np.array(rows, dtype=np.float32)


def cyp1a2_feature_matrix(mols: list) -> "np.ndarray":
    """(N, 3): aromatic_fraction, n_aromatic_rings, pbf (sentinel 10.0 if absent --
    a high value, since PBF near 0 means flat/good-fit-for-CYP1A2)."""
    rows = []
    for mol in mols:
        if mol is None:
            rows.append([0.0, 0.0, 10.0])
            continue
        f = cyp1a2_features(mol)
        pbf = f["pbf"]
        rows.append([f["aromatic_fraction"], float(f["n_aromatic_rings"]), 10.0 if np.isnan(pbf) else pbf])
    return np.array(rows, dtype=np.float32)
