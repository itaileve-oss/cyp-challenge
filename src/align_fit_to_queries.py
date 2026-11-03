"""Align each fit molecule (our SMILES) onto its best-matching query pose (real
CYP3A4-bound inhibitor) via RDKit O3A (shape + MMFF-charge pharmacophore
superposition), then extract the same structural features validated as meaningful
in analyze_query_poses.py (heme-Fe distance/element, approach angle, hotspot-residue
contacts, H-bond flag) from the resulting aligned pose.

Steps per fit molecule:
  1. 2D Morgan-fingerprint similarity to all query ligands -> take the top-K most
     similar queries as alignment candidates (aligning against a very dissimilar
     query would be meaningless).
  2. For every (conformer, candidate query) pair, run O3A and score it; keep the
     single best-scoring alignment.
  3. Apply that alignment's transform to the winning conformer, then compute the
     same pose-derived structural features as the query-pose analysis.
  4. Record the O3A score and 2D similarity to the matched query as CONFIDENCE
     features too -- low similarity/low O3A score should let the model learn to
     trust a given fit molecule's pose features less (important for e.g. CYP1A2
     later, where only 1 query structure exists at all).
"""

import pickle
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from scipy.spatial.distance import cdist

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_K_QUERIES = 3

CONTACT_RADIUS = 4.5
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
POLAR_RESIDUES = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "ASP", "GLU", "LYS", "ARG", "CYS"}
HOTSPOT_RESIDUES = [
    ("A", "ARG", 212), ("A", "SER", 119), ("A", "ILE", 369), ("A", "THR", 309),
    ("A", "ALA", 305), ("A", "ILE", 301), ("A", "PHE", 213), ("A", "ARG", 105),
    ("A", "PHE", 108), ("A", "ALA", 370), ("A", "PHE", 215), ("A", "PHE", 241),
    ("A", "ILE", 120), ("A", "PHE", 304),
]


def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.RemoveHs(mol), radius=2, nBits=2048)


def best_alignment(fit_mol: Chem.Mol, candidate_queries: list[tuple[tuple, Chem.Mol, float]]):
    """Try every (fit conformer) x (candidate query) pair, return the best-scoring
    O3A alignment applied to a copy of fit_mol (single winning conformer only).
    """
    best = None  # (score, query_key, similarity, aligned_single_conf_mol)
    try:
        fit_props = AllChem.MMFFGetMoleculeProperties(fit_mol)
    except Exception:
        return None
    if fit_props is None:
        return None

    for query_key, query_mol, similarity in candidate_queries:
        try:
            query_props = AllChem.MMFFGetMoleculeProperties(query_mol)
        except Exception:
            continue
        if query_props is None:
            continue
        for conf in fit_mol.GetConformers():
            cid = conf.GetId()
            try:
                o3a = rdMolAlign.GetO3A(fit_mol, query_mol, fit_props, query_props, prbCid=cid)
                score = o3a.Score()
            except Exception:
                continue
            if best is None or score > best[0]:
                # Materialize a single-conformer mol with the alignment applied,
                # so downstream code doesn't need to track which conformer won.
                aligned = Chem.Mol(fit_mol, confId=cid)
                aligned.RemoveAllConformers()
                aligned.AddConformer(fit_mol.GetConformer(cid), assignId=True)
                o3a_single = rdMolAlign.GetO3A(aligned, query_mol, fit_props, query_props, prbCid=0)
                o3a_single.Align()
                best = (score, query_key, similarity, aligned)

    return best


def extract_pose_features(mol: Chem.Mol, fe_pos: np.ndarray, plane_normal: np.ndarray,
                           protein_positions: np.ndarray, protein_meta: list) -> dict:
    conf = mol.GetConformer()
    heavy_idx = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() != "H"]
    lig_positions = np.array([conf.GetAtomPosition(i) for i in heavy_idx])
    lig_elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy_idx]

    fe_dists = np.linalg.norm(lig_positions - fe_pos, axis=1)
    best = int(np.argmin(fe_dists))
    best_dist = float(fe_dists[best])
    closest_element = lig_elements[best]
    vec = lig_positions[best] - fe_pos
    vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(vec_norm, plane_normal)), -1, 1))))

    dist_matrix = cdist(lig_positions, protein_positions)
    contact_mask = dist_matrix <= CONTACT_RADIUS
    lig_rows, prot_cols = np.nonzero(contact_mask)
    contacted_residues = set()
    has_hbond = False
    for lig_i, prot_j in zip(lig_rows, prot_cols):
        resname, seqid, chain, atom_name, element = protein_meta[prot_j]
        contacted_residues.add((chain, resname, seqid))
        if dist_matrix[lig_i, prot_j] <= 3.5 and lig_elements[lig_i] in ("N", "O") and element in ("N", "O"):
            has_hbond = True

    n_hydrophobic = sum(1 for (_, r, _) in contacted_residues if r in HYDROPHOBIC_RESIDUES)
    n_polar = sum(1 for (_, r, _) in contacted_residues if r in POLAR_RESIDUES)
    hotspot_hits = sum(1 for h in HOTSPOT_RESIDUES if h in contacted_residues)

    return {
        "fe_distance": best_dist,
        "closest_atom_is_N": closest_element == "N",
        "closest_atom_is_O": closest_element == "O",
        "angle_to_heme_normal": angle,
        "n_contacted_residues": len(contacted_residues),
        "n_hydrophobic_contacts": n_hydrophobic,
        "n_polar_contacts": n_polar,
        "n_hotspot_contacts": hotspot_hits,
        "has_hbond": has_hbond,
    }


def main() -> None:
    from analyze_query_poses import RECEPTOR_PATH, load_receptor_info

    print("Loading receptor info...", flush=True)
    fe_pos_g, plane_normal, protein_positions, protein_meta = load_receptor_info(RECEPTOR_PATH)
    fe_pos = np.array([fe_pos_g.x, fe_pos_g.y, fe_pos_g.z])

    print("Loading query poses...", flush=True)
    with open(PROJECT_ROOT / "data" / "structures" / "cyp3a4_query_mols_aligned.pkl", "rb") as f:
        query_mols: dict[tuple[str, str], Chem.Mol] = pickle.load(f)
    query_items = list(query_mols.items())
    query_fps = [morgan_fp(m) for _, m in query_items]
    print(f"  {len(query_items)} query poses, fingerprints computed", flush=True)

    print("Loading fit-molecule conformer ensembles...", flush=True)
    with open(PROJECT_ROOT / "models" / "fit_conformers_train_cache.pkl", "rb") as f:
        fit_mols: list[Chem.Mol | None] = pickle.load(f)
    print(f"  {len(fit_mols)} fit molecules", flush=True)

    return fe_pos, plane_normal, protein_positions, protein_meta, query_items, query_fps, fit_mols


if __name__ == "__main__":
    main()
