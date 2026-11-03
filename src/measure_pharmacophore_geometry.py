"""Measure the INTRAMOLECULAR geometric relationship between a heme-coordinating
atom and a Ser119-H-bonding atom across the 100 real CYP3A4-bound query poses.

Rationale: global shape/pharmacophore alignment onto a specific reference ligand
failed (median 2D similarity to any of the 100 queries was only 0.19 across our own
compounds -- too dissimilar for a meaningful O3A overlay). But a two-point
geometric pharmacophore is a much weaker, more generalizable constraint: it doesn't
require whole-molecule similarity to any single reference, just SOME pair of atoms
positioned at roughly the empirically-observed relative distance -- the same logic
as the classic CYP2D6 "basic N 5-7 A from the oxidation site" pharmacophore found in
the literature search. Since Fe and Ser119 are fixed landmarks in the common aligned
frame (from align_queries_to_receptor.py), we can measure this directly from real
ligands rather than guessing a target distance.

For each query ligand:
  1. Find its heme-coordinating atom: closest heavy atom to Fe, if within 3.5 A
     (coordination-bond range) AND a plausible lone-pair donor (N, O, or S).
  2. Find its Ser119-H-bonding atom: an N/O ligand atom within 3.5 A of Ser119's
     hydroxyl oxygen (OG).
  3. If a ligand has BOTH, record the INTRAMOLECULAR 3D distance between those two
     specific atoms (not the distance to Fe/Ser119 -- the distance between the two
     ligand atoms themselves, which is what a fit molecule's own conformer can be
     checked against without needing any protein alignment at all).
"""

import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FE_COORD_RADIUS = 3.5
SER119_HBOND_RADIUS = 3.5
LONE_PAIR_DONORS = {"N", "O", "S"}


def find_ser119_og(receptor_path: Path) -> np.ndarray:
    import gemmi

    structure = gemmi.read_structure(str(receptor_path))
    for chain in structure[0]:
        for residue in chain:
            if residue.name == "SER" and residue.seqid.num == 119:
                for atom in residue:
                    if atom.name == "OG":
                        return np.array([atom.pos.x, atom.pos.y, atom.pos.z])
    raise RuntimeError("SER119 OG not found in receptor")


def main() -> None:
    from analyze_query_poses import RECEPTOR_PATH, load_receptor_info

    fe_pos_g, _, _, _ = load_receptor_info(RECEPTOR_PATH)
    fe_pos = np.array([fe_pos_g.x, fe_pos_g.y, fe_pos_g.z])
    ser119_og = find_ser119_og(RECEPTOR_PATH)
    print(f"Fe at {fe_pos.round(1)}, Ser119-OG at {ser119_og.round(1)}, "
          f"Fe-Ser119OG distance = {np.linalg.norm(fe_pos - ser119_og):.2f} A")

    with open(PROJECT_ROOT / "data" / "structures" / "cyp3a4_query_mols_aligned.pkl", "rb") as f:
        query_mols: dict[tuple[str, str], Chem.Mol] = pickle.load(f)

    intramolecular_dists = []
    qualifying = []
    for key, mol in query_mols.items():
        conf = mol.GetConformer()
        heavy_idx = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() != "H"]
        positions = np.array([conf.GetAtomPosition(i) for i in heavy_idx])
        elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy_idx]

        fe_dists = np.linalg.norm(positions - fe_pos, axis=1)
        coord_candidates = [
            i for i in range(len(heavy_idx))
            if fe_dists[i] <= FE_COORD_RADIUS and elements[i] in LONE_PAIR_DONORS
        ]
        if not coord_candidates:
            continue
        coord_atom = coord_candidates[int(np.argmin(fe_dists[coord_candidates]))]

        ser_dists = np.linalg.norm(positions - ser119_og, axis=1)
        hbond_candidates = [
            i for i in range(len(heavy_idx))
            if ser_dists[i] <= SER119_HBOND_RADIUS and elements[i] in ("N", "O") and i != coord_atom
        ]
        if not hbond_candidates:
            continue
        hbond_atom = hbond_candidates[int(np.argmin(ser_dists[hbond_candidates]))]

        intramolecular_d = float(np.linalg.norm(positions[coord_atom] - positions[hbond_atom]))
        intramolecular_dists.append(intramolecular_d)
        qualifying.append((key, elements[coord_atom], elements[hbond_atom], round(intramolecular_d, 2)))

    print(f"\n{len(qualifying)}/{len(query_mols)} query ligands have BOTH a Fe-coordinating atom "
          f"AND a Ser119-H-bonding atom")
    for key, coord_el, hbond_el, d in qualifying:
        print(f"  {key}: coord={coord_el}, hbond_partner={hbond_el}, intramolecular_dist={d} A")

    if intramolecular_dists:
        arr = np.array(intramolecular_dists)
        print(f"\nIntramolecular coord-atom <-> Ser119-hbond-atom distance (A): "
              f"mean={arr.mean():.2f} std={arr.std():.2f} min={arr.min():.2f} max={arr.max():.2f} "
              f"median={np.median(arr):.2f}")


if __name__ == "__main__":
    main()
