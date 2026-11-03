"""Independent GNINA docking pose for each ligand, in the Boltz frame.

Docking is a second, physics-based opinion on where the ligand sits - lesson 1
of the PXR post-mortem: co-folding beats docking, so docking is used as a
cross-check and a source of an alternative pose, never as the primary method.

Design choices that matter:

* The receptor is the ligand's OWN Boltz cluster-medoid prediction, with the
  ligand chain stripped and the heme kept (renamed HEM). Docking into that
  exact receptor puts the docked pose in the SAME coordinate frame as the Boltz
  poses, so a cross-method RMSD is directly meaningful - that agreement is the
  independent-method consensus signal the top PXR teams relied on. The receptor
  is also nearly ligand-independent here (1.44 A CA RMSD between ligands), so
  this is not over-fitting a per-ligand pocket.
* The autobox is built from the HEME, not from the Boltz ligand pose, so the
  search is centred on the active site without being biased toward the answer
  Boltz already gave.
* The ligand 3D structure is embedded from the VERBATIM challenge SMILES - no
  re-protonation or re-tautomerisation, exactly as for co-folding: the
  validator checks connectivity against that string, and the azole/pyridine
  ligands need a neutral ring N lone pair to reach the iron.

Known limitation, measured earlier in this project: Vina-type scoring does not
model heme-Fe coordination (only ~3% of docked poses coordinate, against ~83%
of real inhibitors), so for the 10 type-II ligands the docked pose is expected
to miss the iron. That is informative, not a bug - it is why docking is a
cross-check here.

Usage:
    python dock_pool.py ANALYSIS_DIR OUT_DIR [--only MOL] [--exhaustiveness N]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
RDLogger.DisableLog("rdApp.*")


def split_receptor(pose_pdb: Path, rec_out: Path, heme_out: Path) -> None:
    """Receptor = protein + heme (chain C dropped); heme also written for the autobox."""
    st = gemmi.read_structure(str(pose_pdb))
    st.setup_entities()
    st.remove_hydrogens()
    heme = gemmi.Structure(); heme.add_model(gemmi.Model("1"))
    keep = gemmi.Structure(); keep.add_model(gemmi.Model("1"))
    for ch in st[0]:
        if ch.name == "C":
            continue
        new = gemmi.Chain(ch.name)
        for res in ch:
            res.name = "HEM" if ch.name == "B" else res.name
            new.add_residue(res)
        keep[0].add_chain(new)
        if ch.name == "B":
            hc = gemmi.Chain("B")
            for res in ch:
                hc.add_residue(res)
            heme[0].add_chain(hc)
    keep.setup_entities()
    keep.write_pdb(str(rec_out))
    heme.setup_entities()
    heme.write_pdb(str(heme_out))


def embed_ligand(smiles: str, out_sdf: Path, seed: int = 42) -> None:
    """3D conformer from the verbatim challenge SMILES (no protonation changes)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, str(out_sdf))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--only")
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--num_modes", type=int, default=20)
    ap.add_argument("--autobox_add", type=float, default=8.0)
    ap.add_argument("--cnn", default="rescore", help="gnina --cnn_scoring")
    args = ap.parse_args()

    d = Path(args.analysis_dir)
    out = Path(args.out_dir)
    for sub in ("receptors", "ligands", "docked", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(d / "selected_poses_clustered.csv")
    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    if args.only:
        sel = sel[sel.Molecule_Name == args.only]

    rows = []
    for _, r in sel.iterrows():
        mid = r["Molecule_Name"]
        rec = out / "receptors" / f"{mid}_rec.pdb"
        heme = out / "receptors" / f"{mid}_heme.pdb"
        lig = out / "ligands" / f"{mid}.sdf"
        docked = out / "docked" / f"{mid}_docked.sdf"
        log = out / "logs" / f"{mid}.log"

        split_receptor(Path(r["sub_pdb"]), rec, heme)
        embed_ligand(smi[mid], lig)

        cmd = ["gnina", "--receptor", str(rec), "--ligand", str(lig),
               "--autobox_ligand", str(heme), "--autobox_add", str(args.autobox_add),
               "--out", str(docked), "--log", str(log),
               "--num_modes", str(args.num_modes),
               "--exhaustiveness", str(args.exhaustiveness),
               "--cnn_scoring", args.cnn, "--seed", "42"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        ok = proc.returncode == 0 and docked.exists()
        if not ok:
            print(f"[FAIL] {mid}: {proc.stderr.strip()[:300]}", file=sys.stderr)
            rows.append(dict(Molecule_Name=mid, docked=False))
            continue

        suppl = [m for m in Chem.SDMolSupplier(str(docked), sanitize=False) if m is not None]
        fe = None
        st = gemmi.read_structure(str(heme))
        for ch in st[0]:
            for res in ch:
                for atom in res:
                    if atom.element == gemmi.Element("Fe"):
                        fe = np.array(atom.pos.tolist())
        for rank, m in enumerate(suppl, 1):
            xyz = m.GetConformer().GetPositions()
            heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != "H"]
            dfe = np.linalg.norm(xyz[heavy] - fe, axis=1)
            i = heavy[int(dfe.argmin())]
            rows.append(dict(
                Molecule_Name=mid, docked=True, rank=rank,
                affinity=float(m.GetProp("minimizedAffinity")) if m.HasProp("minimizedAffinity") else np.nan,
                cnn_pose=float(m.GetProp("CNNscore")) if m.HasProp("CNNscore") else np.nan,
                cnn_affinity=float(m.GetProp("CNNaffinity")) if m.HasProp("CNNaffinity") else np.nan,
                min_fe_dist=float(dfe.min()),
                fe_atom=m.GetAtomWithIdx(i).GetSymbol(),
                fe_coordinated=bool(dfe.min() < 2.6),
                sdf=str(docked),
            ))
        print(f"[ok] {mid}: {len(suppl)} modes, best affinity "
              f"{rows[-len(suppl)]['affinity']:.2f}, min Fe {min(x['min_fe_dist'] for x in rows[-len(suppl):]):.2f} A")

    df = pd.DataFrame(rows)
    df.to_csv(out / "docking_results.csv", index=False)
    if "rank" in df:
        top = df[df["rank"] == 1]
        print(f"\n{len(top)} ligands docked | top-mode Fe coordination: "
              f"{int(top.fe_coordinated.sum())}/{len(top)} | "
              f"any-mode coordination: {df.groupby('Molecule_Name').fe_coordinated.any().sum()}/"
              f"{df.Molecule_Name.nunique()}")


if __name__ == "__main__":
    main()
