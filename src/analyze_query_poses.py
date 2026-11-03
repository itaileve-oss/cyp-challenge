"""Empirically characterize what real CYP3A4 inhibitors' bound poses look like,
using the 100 aligned query structures + prepared receptor. Computed BEFORE
deciding what pose-derived features to score our own compounds on -- ground the
feature design in what's actually observed across real inhibitors rather than
guessing.

Looks at:
  - Heme-iron proximity: closest ligand atom to Fe, its element (a heteroatom
    donating a lone pair to Fe is the classic type-II inhibitor mechanism --
    azole antifungals etc.) and approach angle relative to the porphyrin plane.
  - Protein contacts: which residues are contacted, split into hydrophobic vs.
    polar/H-bond-capable, to find "hotspot" pocket residues across the 100 ligands
    (not just heme-proximal chemistry -- distal hydrophobic subpockets matter too
    for CYP3A4's large, multi-lobed active site).
  - Potential H-bond contacts: ligand N/O atoms within H-bonding distance of
    protein N/O atoms.

Receptor and aligned query poses are already in the same coordinate frame by
construction (both built from/aligned onto 4NY4's native deposited coordinates,
and pdbfixer only ADDS atoms -- it never moves existing ones -- so no extra
alignment step is needed here).
"""

import pickle
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECEPTOR_PATH = PROJECT_ROOT / "data" / "receptors" / "cyp3a4_receptor_prepared.pdb"
QUERY_MOLS_PATH = PROJECT_ROOT / "data" / "structures" / "cyp3a4_query_mols_aligned.pkl"

CONTACT_RADIUS = 4.5  # A, general protein-ligand contact
HBOND_RADIUS = 3.5  # A, N/O...N/O potential hydrogen bond
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
POLAR_RESIDUES = {"SER", "THR", "TYR", "ASN", "GLN", "HIS", "ASP", "GLU", "LYS", "ARG", "CYS"}


def load_receptor_info(path: Path):
    structure = gemmi.read_structure(str(path))
    model = structure[0]

    fe_pos = None
    porphyrin_n_positions = []
    protein_positions = []  # (N, 3)
    protein_meta = []  # (resname, seqid, chain, atom_name, element), parallel to protein_positions

    for chain in model:
        for residue in chain:
            if residue.name == "HEM":
                for atom in residue:
                    if atom.element.name == "Fe":
                        fe_pos = atom.pos
                    elif atom.name in ("NA", "NB", "NC", "ND"):
                        porphyrin_n_positions.append(atom.pos)
            elif residue.name not in ("HOH", "HEM"):
                for atom in residue:
                    if atom.element.name == "H":
                        continue
                    protein_positions.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    protein_meta.append((residue.name, residue.seqid.num, chain.name, atom.name, atom.element.name))

    if fe_pos is None or len(porphyrin_n_positions) != 4:
        raise RuntimeError(f"Could not find HEM Fe + 4 pyrrole N atoms (found {len(porphyrin_n_positions)} N)")

    # Best-fit plane normal through the 4 pyrrole N atoms via SVD.
    pts = np.array([[p.x, p.y, p.z] for p in porphyrin_n_positions])
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    plane_normal = vh[2]  # smallest-variance direction = plane normal

    protein_positions = np.array(protein_positions)  # (N, 3)
    return fe_pos, plane_normal, protein_positions, protein_meta


def analyze_one_ligand(
    mol: Chem.Mol,
    fe_pos: gemmi.Position,
    plane_normal: np.ndarray,
    protein_positions: np.ndarray,
    protein_meta: list,
):
    conf = mol.GetConformer()
    fe_np = np.array([fe_pos.x, fe_pos.y, fe_pos.z])

    heavy_idx = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() != "H"]
    lig_positions = np.array([conf.GetAtomPosition(i) for i in heavy_idx])  # (M, 3)
    lig_elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy_idx]

    # Heme-iron proximity + approach angle
    fe_dists = np.linalg.norm(lig_positions - fe_np, axis=1)
    best = int(np.argmin(fe_dists))
    best_dist = float(fe_dists[best])
    closest_element = lig_elements[best]
    vec_to_atom = lig_positions[best] - fe_np
    vec_to_atom_norm = vec_to_atom / (np.linalg.norm(vec_to_atom) + 1e-9)
    angle_to_normal = np.degrees(np.arccos(np.clip(abs(np.dot(vec_to_atom_norm, plane_normal)), -1, 1)))

    # Protein contacts -- vectorized (M ligand heavy atoms) x (N protein heavy atoms)
    # distance matrix via scipy.cdist instead of a nested Python loop (~60 x ~4000
    # pairs x 100 ligands would be very slow atom-by-atom).
    dist_matrix = cdist(lig_positions, protein_positions)  # (M, N)

    contacted_residues = set()
    hbond_contacts = []
    contact_mask = dist_matrix <= CONTACT_RADIUS
    lig_rows, prot_cols = np.nonzero(contact_mask)
    for lig_i, prot_j in zip(lig_rows, prot_cols):
        resname, seqid, chain, atom_name, element = protein_meta[prot_j]
        contacted_residues.add((chain, resname, seqid))
        d = dist_matrix[lig_i, prot_j]
        if d <= HBOND_RADIUS and lig_elements[lig_i] in ("N", "O") and element in ("N", "O"):
            hbond_contacts.append((chain, resname, seqid, atom_name, round(float(d), 2)))

    n_hydrophobic = sum(1 for (_, resname, _) in contacted_residues if resname in HYDROPHOBIC_RESIDUES)
    n_polar = sum(1 for (_, resname, _) in contacted_residues if resname in POLAR_RESIDUES)

    return {
        "fe_distance": round(float(best_dist), 2),
        "closest_atom_element": closest_element,
        "angle_to_heme_normal": round(float(angle_to_normal), 1),
        "n_contacted_residues": len(contacted_residues),
        "n_hydrophobic_contacts": n_hydrophobic,
        "n_polar_contacts": n_polar,
        "contacted_residues": contacted_residues,
        "hbond_contacts": hbond_contacts,
    }


def main() -> None:
    print("Loading receptor...", flush=True)
    fe_pos, plane_normal, protein_positions, protein_meta = load_receptor_info(RECEPTOR_PATH)
    print(f"  Heme Fe at ({fe_pos.x:.1f}, {fe_pos.y:.1f}, {fe_pos.z:.1f}), "
          f"{len(protein_meta)} protein heavy atoms", flush=True)

    with open(QUERY_MOLS_PATH, "rb") as f:
        query_mols: dict[tuple[str, str], Chem.Mol] = pickle.load(f)
    print(f"Analyzing {len(query_mols)} query poses...", flush=True)

    results = {}
    residue_contact_counter: Counter = Counter()
    for key, mol in query_mols.items():
        try:
            r = analyze_one_ligand(mol, fe_pos, plane_normal, protein_positions, protein_meta)
            results[key] = r
            for res in r["contacted_residues"]:
                residue_contact_counter[res] += 1
        except Exception as e:
            print(f"  FAILED {key}: {e}")

    print(f"\n=== Summary across {len(results)} real CYP3A4 inhibitor poses ===\n")

    fe_dists = sorted(r["fe_distance"] for r in results.values())
    print(f"Heme-Fe distance (closest ligand atom), A: "
          f"min={fe_dists[0]:.2f} median={fe_dists[len(fe_dists)//2]:.2f} max={fe_dists[-1]:.2f}")
    print(f"  Ligands with an atom within 3.5 A of Fe (candidate Fe-coordinating): "
          f"{sum(1 for d in fe_dists if d <= 3.5)}/{len(fe_dists)}")
    print(f"  Ligands with an atom within 5.0 A of Fe: {sum(1 for d in fe_dists if d <= 5.0)}/{len(fe_dists)}")

    closest_elements = Counter(r["closest_atom_element"] for r in results.values())
    print(f"\nElement closest to Fe (across all {len(results)} ligands): {dict(closest_elements.most_common())}")

    angles = sorted(r["angle_to_heme_normal"] for r in results.values())
    print(f"\nApproach angle to heme-plane normal, deg (0=perpendicular/face-on, 90=in-plane/edge-on): "
          f"min={angles[0]:.0f} median={angles[len(angles)//2]:.0f} max={angles[-1]:.0f}")

    n_contacts = sorted(r["n_contacted_residues"] for r in results.values())
    n_hydrophobic = sorted(r["n_hydrophobic_contacts"] for r in results.values())
    n_polar = sorted(r["n_polar_contacts"] for r in results.values())
    print(f"\nContacted residues per ligand: min={n_contacts[0]} median={n_contacts[len(n_contacts)//2]} max={n_contacts[-1]}")
    print(f"  hydrophobic: median={n_hydrophobic[len(n_hydrophobic)//2]}, polar: median={n_polar[len(n_polar)//2]}")

    print(f"\nTop 20 hotspot residues (contacted by most of the {len(results)} ligands):")
    for (chain, resname, seqid), count in residue_contact_counter.most_common(20):
        kind = "hydrophobic" if resname in HYDROPHOBIC_RESIDUES else ("polar" if resname in POLAR_RESIDUES else "other")
        print(f"  {resname}{seqid} (chain {chain}, {kind}): {count}/{len(results)} ligands ({100*count/len(results):.0f}%)")

    n_with_hbond = sum(1 for r in results.values() if r["hbond_contacts"])
    print(f"\nLigands with >=1 potential H-bond contact (N/O within {HBOND_RADIUS} A): {n_with_hbond}/{len(results)}")

    out_path = PROJECT_ROOT / "data" / "structures" / "cyp3a4_query_pose_analysis.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nDetailed per-ligand results saved to {out_path}")


if __name__ == "__main__":
    main()
