"""Two-point CYP3A4 heme-coordination / Ser119-H-bond geometric pharmacophore
feature.

Derived empirically in measure_pharmacophore_geometry.py: 55/100 real CYP3A4-bound
inhibitors have BOTH a heme-coordinating nitrogen (within 3.5 A of Fe) and a
Ser119-H-bonding N/O atom (within 3.5 A of Ser119's OG) -- and the INTRAMOLECULAR
distance between those two specific atoms is tightly distributed: 6.79 +/- 1.07 A
across all 55.

Why this works where whole-molecule O3A alignment didn't: our own compounds are
chemically too dissimilar to any single one of the 100 known ligands (median 2D
Tanimoto only 0.19) for a shape/pharmacophore overlay to mean anything. But this is
a much weaker constraint -- just two atoms at roughly the right relative distance,
not overall shape similarity -- so structurally unrelated chemotypes can satisfy it
independently, the same way the classic CYP2D6 "basic N 5-7 A from the oxidation
site" pharmacophore works across very diverse substrate chemotypes. And critically,
it's purely INTRAMOLECULAR: no alignment to the protein or to any reference ligand
is needed at all, just the fit molecule's own conformer ensemble.

CALIBRATION CAVEAT (checked before trusting this): searching "any of up to 20
conformers x any candidate atom pair" for a match against a fixed target distance
suffers a look-elsewhere effect -- arbitrary control targets (4.0, 10.0 A, nothing
to do with CYP3A4) got 71%/38% "match" rates too, meaning the naive version mostly
just detects "this is a typical drug-sized molecule with several heteroatoms," not
real pharmacophore signal. Restricting to the single LOWEST-ENERGY conformer (not
all 20) meaningfully sharpens this (65%/48%/21% for targets 4.0/6.79/10.0 -- more
separation, though still not perfectly clean) and is also the more chemically
defensible choice regardless (only a genuinely accessible low-energy conformation
should count). The continuous best-matching distance is treated as the primary
signal for the model to weigh, rather than imposing an a-priori "match" cutoff.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

TARGET_DIST = 6.79
TARGET_STD = 1.07

# Pyridine/imidazole-type aromatic N (available lone pair, not the pyrrole-type NH
# whose lone pair is delocalized into the aromatic ring).
_COORD_AROMATIC_N = Chem.MolFromSmarts("[n;H0;+0]")
# Neutral (unprotonated) primary/secondary/tertiary amine N, excluding amide/urea/
# guanidine-type N where the lone pair is delocalized into a C=O or C=N and not
# available for heme coordination. Formal charge 0 matters here specifically
# because our fit molecules are already protonated at pH 7.4 (prepare_fit_ligands.py)
# -- a basic amine that picked up a proton is now genuinely unavailable to
# coordinate Fe, not just excluded for convenience.
_COORD_AMINE_N = Chem.MolFromSmarts("[NX3;H0,H1,H2;+0;!$(NC=O);!$(N-C=[N,S])]")
_HBOND_ATOM = Chem.MolFromSmarts("[#7,#8]")


def _lowest_energy_conf_id(mol: Chem.Mol, props=None) -> int:
    if props is None:
        props = AllChem.MMFFGetMoleculeProperties(mol)
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    if props is None:
        return conf_ids[0]
    best_id, best_e = conf_ids[0], None
    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        if ff is None:
            continue
        e = ff.CalcEnergy()
        if best_e is None or e < best_e:
            best_e, best_id = e, cid
    return best_id


def pharmacophore_features(mol: Chem.Mol) -> dict:
    coord_idx = {m[0] for m in mol.GetSubstructMatches(_COORD_AROMATIC_N)}
    coord_idx |= {m[0] for m in mol.GetSubstructMatches(_COORD_AMINE_N)}
    hbond_idx = {m[0] for m in mol.GetSubstructMatches(_HBOND_ATOM)}

    has_coord = len(coord_idx) > 0
    has_hbond = len(hbond_idx) > 0
    result = {
        "has_coord_atom": has_coord,
        "has_hbond_atom": has_hbond,
        "pharmacophore_best_dist": np.nan,
        "pharmacophore_deviation": np.nan,
        "pharmacophore_match": False,
    }
    if not (has_coord and has_hbond) or mol.GetNumConformers() == 0:
        return result

    coord_list = list(coord_idx)
    hbond_list = list(hbond_idx)
    cid = _lowest_energy_conf_id(mol)
    pos = mol.GetConformer(cid).GetPositions()
    coord_pos = pos[coord_list]  # (C, 3)
    hbond_pos = pos[hbond_list]  # (H, 3)
    dmat = np.linalg.norm(coord_pos[:, None, :] - hbond_pos[None, :, :], axis=2)  # (C, H)
    best_dist = None if dmat.size == 0 else float(dmat.flat[np.argmin(np.abs(dmat - TARGET_DIST))])

    if best_dist is None:
        return result

    deviation = abs(best_dist - TARGET_DIST)
    result.update({
        "pharmacophore_best_dist": best_dist,
        "pharmacophore_deviation": deviation,
        "pharmacophore_match": deviation <= TARGET_STD,
    })
    return result


def pharmacophore_feature_matrix(mols: list) -> "np.ndarray":
    """(N, 4) numeric matrix: has_coord_atom, has_hbond_atom, pharmacophore_best_dist
    (NaN-filled with a sentinel far outside the target range if absent), pharmacophore_match.
    Molecules that are None (failed fit-molecule prep) get an all-absent row.
    """
    rows = []
    for mol in mols:
        if mol is None:
            rows.append([0.0, 0.0, 20.0, 0.0])
            continue
        f = pharmacophore_features(mol)
        dist = f["pharmacophore_best_dist"]
        rows.append([
            float(f["has_coord_atom"]),
            float(f["has_hbond_atom"]),
            20.0 if np.isnan(dist) else dist,  # sentinel: far from any real target
            float(f["pharmacophore_match"]),
        ])
    return np.array(rows, dtype=np.float32)
