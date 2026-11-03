"""GalaxyDock2-HEME docking of the challenge ligands into their Boltz receptors.

GalaxyDock2-HEME is the right docking engine for a P450: unlike Vina/GNINA it
models heme-iron coordination explicitly, so it can reproduce the type-II
binding mode that Vina-type scoring misses (measured earlier in this project:
~3% of Vina-docked poses coordinate the iron, against ~83% of real inhibitors;
a GNINA test run here found 0/20 coordinating modes for a ligand whose Boltz
poses coordinate in 20/20).

Protocol follows the project's established GD2-HEME setup (src/run_batch_ensemble.py
and src/launch_test_docking.sh), with three structure-track adaptations:

* RECEPTOR is each ligand's own Boltz cluster-medoid prediction rather than a
  crystal structure, so docked poses land in the SAME coordinate frame as that
  ligand's Boltz poses and a cross-method RMSD is directly meaningful. The heme
  is moved into chain A as residue 601 (resname HEM, HETATM) because
  preprocess_protein() only keeps heme found in the protein's own chain.
* BOX CENTRE is computed per receptor instead of being a fixed aligned-frame
  constant: every Boltz sample has its own rigid-body frame, so no shared
  centre exists. The centre is placed on the heme-iron distal normal, offset
  into the substrate cavity, which reproduces the "reference-ligand centroid"
  convention geometrically without biasing toward Boltz's own answer.
* LIGANDS are embedded from the VERBATIM challenge SMILES with no dimorphite
  protonation step. This differs deliberately from the activity-track prep
  (prepare_ligands_gd2heme_v2.py protonates at pH 7.4): the submission
  validator checks connectivity against the exact challenge SMILES, and an
  imidazole protonated at pH 7.4 loses the neutral lone pair it needs to
  coordinate the iron.

Run directories live under short absolute paths (~/gd2r/<MOL>) because
GD2-HEME parses fixed-length path fields and fails on long ones.

Usage:
    python dock_gd2heme_structure.py ANALYSIS_DIR [--only MOL] [--workers N] [--seed N]
"""
import argparse
import os
import shutil
import subprocess as sp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parents[2]
GD2_HOME = ROOT / "external/GalaxyDock2_HEME/GalaxyDock2_HEME"
CHIMERA_BIN = Path.home() / "tools/UCSF-Chimera64/bin"
OBABEL = Path.home() / "miniconda3/envs/GNINA/bin/obabel"
GD2_LINK = Path.home() / "g2"          # short symlink: long paths break GD2
RUN_ROOT = Path.home() / "gd2r"
HEME_RES_NUM = 601
DISTAL_OFFSET = 5.0                    # A along the distal normal, into the cavity
TIMEOUT_S = 1800
RDLogger.DisableLog("rdApp.*")


def build_receptor(pose_pdb: Path, out_pdb: Path) -> np.ndarray:
    """Protein + heme in ONE chain (A, heme as HETATM HEM 601); returns box centre."""
    st = gemmi.read_structure(str(pose_pdb))
    st.remove_hydrogens()
    model = st[0]

    prot_lines, heme_atoms, fe, sg = [], [], None, None
    heme_xyz = []
    for ch in model:
        for res in ch:
            for atom in res:
                xyz = np.array(atom.pos.tolist())
                if ch.name == "A":
                    if res.seqid.num == 442 and atom.name == "SG":
                        sg = xyz
                elif ch.name == "B":
                    heme_xyz.append(xyz)
                    heme_atoms.append((atom.name, atom.element.name, xyz))
                    if atom.element == gemmi.Element("Fe"):
                        fe = xyz

    # keep protein chain A only, drop the ligand (chain C) and the heme copy
    keep = gemmi.Structure()
    keep.add_model(gemmi.Model("1"))
    new = gemmi.Chain("A")
    for res in model["A"]:
        new.add_residue(res)
    keep[0].add_chain(new)
    keep.setup_entities()
    lines = [ln for ln in keep.make_pdb_string().splitlines()
             if ln.startswith(("ATOM", "TER"))]

    # Strict PDB columns: 13-16 atom name, 17 altLoc, 18-20 resName, 22 chainID,
    # 23-26 resSeq. Getting these off by one puts HEM in the altLoc field and
    # GD2-HEME's preprocess_protein() then reports the heme as missing.
    serial = 9000
    for name, elem, xyz in heme_atoms:
        serial += 1
        atom_name = name if len(name) >= 4 else f" {name:<3}"
        lines.append(
            f"HETATM{serial:>5} {atom_name:<4} HEM A{HEME_RES_NUM:>4}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem.upper():>2}"
        )
    lines.append("END")
    out_pdb.write_text("\n".join(lines) + "\n")

    heme = np.array(heme_xyz)
    centred = heme - heme.mean(axis=0)
    normal = np.linalg.svd(centred)[2][-1]
    # point the normal AWAY from the proximal cysteine, i.e. into the substrate cavity
    if (sg - fe) @ normal > 0:
        normal = -normal
    return fe + DISTAL_OFFSET * normal


def build_ligand(smiles: str, out_mol2: Path, seed: int = 42) -> None:
    """3D mol2 from the verbatim challenge SMILES - no protonation change."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    sdf = out_mol2.with_suffix(".sdf")
    Chem.MolToMolFile(mol, str(sdf))
    sp.run([str(OBABEL), str(sdf), "-O", str(out_mol2)], check=True,
           stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def dock_one(job: tuple) -> tuple:
    mid, pose_pdb, smiles, seed = job
    run_dir = RUN_ROOT / mid
    run_dir.mkdir(parents=True, exist_ok=True)
    info = run_dir / "GD2_HEME_fb.E.info"
    if info.exists():
        return mid, True, "already done"

    try:
        centre = build_receptor(Path(pose_pdb), run_dir / "receptor.pdb")
        build_ligand(smiles, run_dir / "ligand.mol2", seed)
    except Exception as exc:  # noqa: BLE001
        return mid, False, f"prep failed: {exc}"

    env = os.environ.copy()
    env["PATH"] = f"{env.get('PATH','')}:{CHIMERA_BIN}"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"

    cmd = ["python3", str(GD2_LINK / "script/run_GalaxyDock2_heme.py"),
           "-d", str(GD2_LINK), "-p", "receptor.pdb", "-l", "ligand.mol2",
           "-x", f"{centre[0]:.3f}", "-y", f"{centre[1]:.3f}", "-z", f"{centre[2]:.3f}",
           "--heme_res_num", str(HEME_RES_NUM), "--chain", "A",
           "--random_seed", str(seed)]
    (run_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")
    try:
        res = sp.run(cmd, cwd=str(run_dir), env=env, stdout=sp.PIPE,
                     stderr=sp.STDOUT, text=True, timeout=TIMEOUT_S)
    except sp.TimeoutExpired:
        (run_dir / "failure.log").write_text("TIMEOUT")
        return mid, False, "timeout"
    if not info.exists():
        (run_dir / "failure.log").write_text(res.stdout[-6000:])
        return mid, False, "no output"
    return mid, True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir")
    ap.add_argument("--only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not GD2_LINK.exists():
        GD2_LINK.symlink_to(GD2_HOME)
    RUN_ROOT.mkdir(exist_ok=True)

    d = Path(args.analysis_dir)
    sel = pd.read_csv(d / "selected_poses_clustered.csv")
    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    if args.only:
        sel = sel[sel.Molecule_Name == args.only]

    jobs = [(r["Molecule_Name"], r["sub_pdb"], smi[r["Molecule_Name"]], args.seed)
            for _, r in sel.iterrows()]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for mid, ok, msg in ex.map(dock_one, jobs):
            print(f"[{'ok' if ok else 'FAIL'}] {mid} {msg}", flush=True)
            results.append(dict(Molecule_Name=mid, ok=ok, msg=msg,
                                run_dir=str(RUN_ROOT / mid)))
    out = pd.DataFrame(results)
    out.to_csv(d / "gd2heme_runs.csv", index=False)
    print(f"\n{int(out.ok.sum())}/{len(out)} docked -> {RUN_ROOT}")


if __name__ == "__main__":
    main()
