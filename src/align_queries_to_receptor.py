"""Superimpose every query structure's protein chain onto the reference receptor
(4NY4) and carry that same rigid-body transform over to its bound ligand's
coordinates.

Why this is needed: the 100 CYP3A4 query structures (see prepare_query_structures.py)
were each solved independently, in their own crystal's coordinate frame -- their
ligand poses are NOT directly comparable/poolable until every structure has been
superimposed into one common frame. Uses gemmi's native `calculate_superposition`
(C-alpha-based rigid-body fit) rather than adding a Biopython dependency for the
same job.

Chain-matching note: a ligand's HETATM chain ID doesn't always match its bound
protein chain's ID depending on deposition convention, so we track which chain the
ligand was actually found in (during extraction) and align that SAME chain -- not
just "the first chain" -- since multi-copy asymmetric units can have small
conformational differences between copies.
"""

import pickle
from pathlib import Path

import gemmi
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CIF_CACHE = PROJECT_ROOT / "data" / "structures" / "raw_cif"
REFERENCE_PDBID = "4NY4"


def load_structure(pdbid: str) -> gemmi.Structure:
    cache_path = RAW_CIF_CACHE / f"{pdbid}.cif"
    if not cache_path.exists():
        import urllib.request

        with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdbid}.cif", timeout=30) as r:
            cache_path.write_bytes(r.read())
    return gemmi.read_structure(str(cache_path))


def find_ligand_chain(structure: gemmi.Structure, comp_id: str) -> str | None:
    """Name of the chain containing the first instance of comp_id."""
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.name == comp_id:
                return chain.name
    return None


def get_polymer_chain(structure: gemmi.Structure, preferred_chain_name: str | None) -> gemmi.ResidueSpan | None:
    """The polymer (protein) ResidueSpan for a chain, preferring preferred_chain_name
    if it has one, else the first polymer-containing chain in the model.
    """
    model = structure[0]
    if preferred_chain_name is not None:
        for chain in model:
            if chain.name == preferred_chain_name:
                polymer = chain.get_polymer()
                if len(polymer) > 0:
                    return polymer
    for chain in model:
        polymer = chain.get_polymer()
        if len(polymer) > 0:
            return polymer
    return None


def main() -> None:
    isoform = "cyp3a4"
    query_mols_path = PROJECT_ROOT / "data" / "structures" / f"{isoform}_query_mols.pkl"
    with open(query_mols_path, "rb") as f:
        query_mols: dict[tuple[str, str], Chem.Mol] = pickle.load(f)
    print(f"Loaded {len(query_mols)} query molecules", flush=True)

    ref_structure = load_structure(REFERENCE_PDBID)
    ref_polymer = get_polymer_chain(ref_structure, preferred_chain_name="A")
    if ref_polymer is None:
        raise RuntimeError(f"No protein chain found in reference structure {REFERENCE_PDBID}")
    print(f"Reference receptor: {REFERENCE_PDBID}, {len(ref_polymer)} polymer residues", flush=True)

    aligned_mols: dict[tuple[str, str], Chem.Mol] = {}
    rmsds: dict[str, float] = {}
    failed = []

    for (pdbid, comp_id), mol in query_mols.items():
        if pdbid.upper() == REFERENCE_PDBID:
            # Reference structure itself needs no transform.
            aligned_mols[(pdbid, comp_id)] = mol
            rmsds[pdbid] = 0.0
            continue

        try:
            structure = load_structure(pdbid)
            ligand_chain_name = find_ligand_chain(structure, comp_id)
            movable_polymer = get_polymer_chain(structure, preferred_chain_name=ligand_chain_name)
            if movable_polymer is None:
                failed.append((pdbid, comp_id, "no polymer chain"))
                continue

            result = gemmi.calculate_superposition(
                ref_polymer, movable_polymer, gemmi.PolymerType.PeptideL, gemmi.SupSelect.CaP
            )
            transform = result.transform  # gemmi.Transform: rotation (Mat33) + translation (Vec3)

            new_mol = Chem.Mol(mol)
            conf = new_mol.GetConformer()
            for i in range(new_mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                new_pos = transform.apply(gemmi.Position(pos.x, pos.y, pos.z))
                conf.SetAtomPosition(i, (new_pos.x, new_pos.y, new_pos.z))

            aligned_mols[(pdbid, comp_id)] = new_mol
            rmsds[pdbid] = result.rmsd
        except Exception as e:
            failed.append((pdbid, comp_id, str(e)))

    print(f"Aligned {len(aligned_mols)}/{len(query_mols)} query poses into {REFERENCE_PDBID}'s frame", flush=True)
    if failed:
        print(f"Failed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")

    rmsd_values = [v for v in rmsds.values() if v > 0]
    if rmsd_values:
        rmsd_values.sort()
        print(
            f"Superposition C-alpha RMSD (A): min={rmsd_values[0]:.2f}, "
            f"median={rmsd_values[len(rmsd_values)//2]:.2f}, max={rmsd_values[-1]:.2f}"
        )

    out_path = PROJECT_ROOT / "data" / "structures" / f"{isoform}_query_mols_aligned.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(aligned_mols, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
