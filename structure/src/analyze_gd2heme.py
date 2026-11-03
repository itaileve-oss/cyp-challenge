"""Analyse the GalaxyDock2-HEME pose pool and compare it with Boltz-2.

The GD2-HEME receptor was built from each ligand's Boltz cluster-medoid pose
and docking never moves the receptor, so the docked poses are already in that
ligand's Boltz frame: the Boltz-vs-GD2 ligand RMSD below needs no superposition
and is directly meaningful. That cross-method agreement is the independent
confirmation this project otherwise lacks (no MLIP dE available locally), and
it is the signal the leading PXR teams built their selection on.

Reads `~/gd2r/<MOL>/GD2_HEME_fb.{mol2,E.info}` and the Boltz medoid picks.

Usage:
    python analyze_gd2heme.py ANALYSIS_DIR [--rundir ~/gd2r]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

ROOT = Path(__file__).resolve().parents[2]
RDLogger.DisableLog("rdApp.*")
COORDINATING_N = Chem.MolFromSmarts("[n;X2;H0]")
FE_COORD_CUTOFF = 2.6


def strip_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Delete H atoms by atomic number.

    Chem.MolFromMol2Block(removeHs=True) leaves polar hydrogens in place when
    sanitize=False, which GD2-HEME output needs. That left the docked pose with
    25 atoms against the template's 23 heavy atoms, so every symmetry-aware
    RMSD returned NaN - an apparent "no agreement" that was purely an atom-count
    mismatch. Strip them explicitly instead of trusting the parser flag.
    """
    rw = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1),
                      reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def flatten(mol: Chem.Mol) -> Chem.Mol:
    """Connectivity-only copy: all bonds single, no aromatic flags, no charges.

    Lets two representations of the same molecule match for RMSD regardless of
    how each source perceived aromaticity. Symmetry handling is unaffected, as
    graph automorphism still finds the equivalent atom mappings; the only cost
    is that a few extra mappings become permissible, which can slightly
    understate RMSD - conservative for an agreement test.
    """
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_SYMMRINGS
                     | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
                     catchErrors=True)
    return m


def read_mol2_poses(path: Path) -> list[Chem.Mol]:
    blocks = path.read_text().split("@<TRIPOS>MOLECULE")
    mols = []
    for b in blocks:
        if not b.strip():
            continue
        m = Chem.MolFromMol2Block("@<TRIPOS>MOLECULE" + b, sanitize=False, removeHs=False)
        if m is not None:
            mols.append(strip_hydrogens(m))
    return mols


def read_scores(path: Path) -> pd.DataFrame:
    rows = []
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 12 and parts[0].isdigit():
            rows.append(dict(rank=int(parts[0]), energy=float(parts[1]),
                             atdk_e=float(parts[3]), int_e=float(parts[4]),
                             ds_e=float(parts[5]), hm_e=float(parts[6]),
                             plp=float(parts[7])))
    return pd.DataFrame(rows)


def fe_from_receptor(path: Path) -> np.ndarray:
    for ln in path.read_text().splitlines():
        if ln.startswith("HETATM") and ln[12:16].strip() == "FE":
            return np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    raise ValueError(f"no FE in {path}")


def retemplate(mol: Chem.Mol, smiles: str) -> Chem.Mol | None:
    """Bond orders from the challenge SMILES, so symmetry-aware RMSD can match.

    Falls back to the organizers' more tolerant matcher, which tries every
    topological match rather than the first one RDKit happens to return.
    """
    template = Chem.MolFromSmiles(smiles)
    try:
        return AllChem.AssignBondOrdersFromTemplate(template, mol)
    except Exception:
        pass
    try:
        import sys
        sys.path.insert(0, str(ROOT / "external/CYP-Challenge-Tutorial"))
        from evaluation.posebusters_utils import assign_bond_orders_from_template
        return assign_bond_orders_from_template(template, mol)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--rundir", default=str(Path.home() / "gd2r"))
    args = ap.parse_args()

    d = Path(args.analysis_dir)
    run = Path(args.rundir).expanduser()
    sel = pd.read_csv(d / "selected_poses_clustered.csv")
    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    feats = pd.read_csv(d / "pose_features.csv").set_index("pose")

    pose_rows, summary = [], []
    for _, r in sel.iterrows():
        mid = r["Molecule_Name"]
        rd = run / mid
        mol2, info = rd / "GD2_HEME_fb.mol2", rd / "GD2_HEME_fb.E.info"
        if not (mol2.exists() and info.exists()):
            print(f"[skip] {mid}: no GD2 output")
            continue

        fe = fe_from_receptor(rd / "receptor.pdb")
        mols = read_mol2_poses(mol2)
        scores = read_scores(info)
        can_coord = bool(Chem.MolFromSmiles(smi[mid]).GetSubstructMatches(COORDINATING_N))

        # Boltz medoid ligand, same frame (receptor came from this pose)
        boltz = Chem.MolFromMolFile(feats.loc[r["pose"], "lig_sdf"], sanitize=False)
        boltz_t = retemplate(boltz, smi[mid])

        boltz_flat = flatten(boltz)
        dists, rmsds = [], []
        for i, m in enumerate(mols):
            xyz = m.GetConformer().GetPositions()
            dfe = np.linalg.norm(xyz - fe, axis=1)
            j = int(dfe.argmin())
            mt = retemplate(m, smi[mid])
            rms = np.nan
            if mt is not None and boltz_t is not None:
                try:
                    rms = float(rdMolAlign.CalcRMS(mt, boltz_t))
                except Exception:
                    rms = np.nan
            if not np.isfinite(rms):
                # Aromaticity is perceived differently from a GD2 mol2 than from
                # a Boltz PDB, so CalcRMS can find no substructure match even
                # with identical atom counts. Compare the flattened topology
                # instead: still symmetry-aware via graph automorphism.
                try:
                    rms = float(rdMolAlign.CalcRMS(flatten(m), boltz_flat))
                except Exception:
                    rms = np.nan
            dists.append(dfe.min())
            rmsds.append(rms)
            pose_rows.append(dict(Molecule_Name=mid, gd2_rank=i + 1,
                                  min_fe_dist=dfe.min(),
                                  fe_atom=m.GetAtomWithIdx(j).GetSymbol(),
                                  fe_coordinated=bool(dfe.min() < FE_COORD_CUTOFF),
                                  rmsd_to_boltz=rms))
        dists, rmsds = np.array(dists), np.array(rmsds)
        best = scores.iloc[0] if len(scores) else {}
        summary.append(dict(
            Molecule_Name=mid, n_gd2=len(mols), can_coordinate=can_coord,
            gd2_coord=int((dists < FE_COORD_CUTOFF).sum()),
            gd2_med_fe=round(float(np.median(dists)), 2),
            boltz_fe=r["min_fe_dist"], boltz_top_frac=r["top_frac"],
            best_energy=round(float(best.get("energy", np.nan)), 2),
            best_hm_e=round(float(best.get("hm_e", np.nan)), 2),
            rmsd_top1=round(float(rmsds[0]), 2) if len(rmsds) else np.nan,
            rmsd_best=round(float(np.nanmin(rmsds)), 2) if np.isfinite(rmsds).any() else np.nan,
            agree_2A=bool(np.nanmin(rmsds) < 2.0) if np.isfinite(rmsds).any() else False,
        ))

    pd.DataFrame(pose_rows).to_csv(d / "gd2heme_poses.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(d / "gd2heme_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(s.to_string(index=False))
    print(f"\ncross-method agreement (some GD2 pose within 2 A of the Boltz pick): "
          f"{int(s.agree_2A.sum())}/{len(s)}")
    print(f"median best-case Boltz-GD2 RMSD: {s.rmsd_best.median():.2f} A | "
          f"median top-ranked GD2 RMSD: {s.rmsd_top1.median():.2f} A")
    coord = s[s.can_coordinate]
    print(f"type-II ligands: GD2 coordinates in {coord.gd2_coord.sum()}/{coord.n_gd2.sum()} poses "
          f"(median Fe {coord.gd2_med_fe.median():.2f} A vs Boltz {coord.boltz_fe.median():.2f} A)")


if __name__ == "__main__":
    main()
