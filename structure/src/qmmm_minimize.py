"""QM/MM geometry optimisation of CYP3A4 poses with sander (semiempirical QM).

Why QM/MM rather than a force field, established by measurement in this project:
  * A fixed-charge FF cannot hold an Fe-N dative bond. Amber + the Li/Merz
    Fe3+ 12-6 LJ puts the Fe-N minimum at 3.21 A, so minimisation PUSHES the
    nitrogen out (crystal median is 2.44 A, our poses 2.05 A - both inside the
    repulsive wall). CHARMM36 fails the opposite way: eps_Fe = 0 and q = +0.24
    give no wall at all and monotonic attraction, so the nitrogen COLLAPSES.
    Either way the answer is set by invented parameters, not by physics.
  * Putting the whole pocket in QM is hopeless (GFN2 on 954 atoms produced no
    energy in 25 min), but a QM/MM *region* of ~100 atoms is cheap - which is
    what makes this tractable.

So: QM = ligand + heme + Cys442 side chain (the coordination chemistry),
MM = the rest of the protein (ff14SB), protein restrained, ligand free.
The Fe-N distance is then decided by electrons, with no restraint target to
guess - and the result can be checked against the 96-structure crystal
distribution as independent validation.

Heme MM parameters: the PENTACOORDINATED set (`hemefe2_pentacoordinated`),
whose 43 heavy-atom names match Boltz's heme exactly and which has no axial
water (the `hem_water` HEW unit carries OH/HO1/HO2 that would clash with the
ligand). Its Fe(II) assignment barely matters because Fe sits in the QM region.

Usage:
    python qmmm_minimize.py POSE_PDB OUT_DIR LIGPARAM_DIR [--theory PM6] [--maxcyc 500]
"""
import argparse
import shutil
import subprocess as sp
from pathlib import Path

import gemmi
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HEME_PREP = ROOT / "structure/params/heme/hemefe2_pentacoordinated/HEM.in"
HEME_PRM = ROOT / "structure/params/heme/hemefe2_pentacoordinated/HEM.prm"
CYS_RESNUM = 442


def split_pose(pdb: Path, d: Path) -> tuple[int, int]:
    """Write protein / heme / ligand separately; returns (n_prot_res, n_lig_atoms)."""
    st = gemmi.read_structure(str(pdb))
    st.remove_hydrogens()
    st.setup_entities()
    model = st[0]

    prot = gemmi.Structure(); prot.add_model(gemmi.Model("1"))
    pc = gemmi.Chain("A")
    heme_lines, lig_lines = [], []
    nres = 0
    for ch in model:
        for res in ch:
            if res.name == "LIG":
                for atom in res:
                    lig_lines.append(atom)
            elif res.name == "HEM":
                for atom in res:
                    heme_lines.append(atom)
            else:
                pc.add_residue(res)
                nres += 1
    prot[0].add_chain(pc)
    prot.setup_entities()
    prot.write_pdb(str(d / "prot_raw.pdb"))

    def write_het(atoms, resname, fn):
        with (d / fn).open("w") as f:
            for i, a in enumerate(atoms, 1):
                nm = a.name if len(a.name) >= 4 else f" {a.name:<3}"
                f.write(f"HETATM{i:>5} {nm:<4} {resname} A   1    "
                        f"{a.pos.x:8.3f}{a.pos.y:8.3f}{a.pos.z:8.3f}  1.00  0.00"
                        f"          {a.element.name.upper():>2}\n")
            f.write("END\n")
    write_het(heme_lines, "HEM", "heme.pdb")
    write_het(lig_lines, "LIG", "lig_pose.pdb")
    return nres, len(lig_lines)


def retemplate_ligand(pose_pdb: Path, template_mol2: Path, smiles: str, out_mol2: Path) -> int:
    """Write a mol2 carrying the TEMPLATE's atom names/types/charges but THIS
    pose's coordinates.

    tleap types the ligand by matching atom names against the parameterised
    unit, and Boltz names its ligand atoms independently of antechamber
    (C35/N34/S44 vs C1/N1/S1), so loading the pose PDB directly gives
    "Atom ... does not have a type -> Failed to generate parameters". The
    parameters are per ligand and pose-independent, so the right move is to
    keep the template file and swap in the new coordinates.

    Hydrogens are rebuilt on the pose with RDKit (Boltz writes heavy atoms
    only) and matched to template hydrogens through their parent heavy atom,
    which is adequate because hydrogens on one heavy atom are interchangeable
    for a minimisation starting point.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    # --- pose geometry, hydrogens rebuilt, bond orders from the challenge SMILES
    st = gemmi.read_structure(str(pose_pdb))
    st.remove_hydrogens()
    block = ["COMPND    LIG"]
    n = 0
    for ch in st[0]:
        for res in ch:
            if res.name != "LIG":
                continue
            for a in res:
                n += 1
                block.append(f"HETATM{n:>5} {a.name:<4} LIG A   1    "
                             f"{a.pos.x:8.3f}{a.pos.y:8.3f}{a.pos.z:8.3f}  1.00  0.00"
                             f"          {a.element.name.upper():>2}")
    block.append("END")
    pose = Chem.MolFromPDBBlock("\n".join(block), removeHs=True, sanitize=False)
    pose = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), pose)
    Chem.SanitizeMol(pose)
    pose = Chem.AddHs(pose, addCoords=True)

    # --- template atom block
    lines = template_mol2.read_text().splitlines()
    sec, atom_idx = "", []
    for i, ln in enumerate(lines):
        if ln.startswith("@<TRIPOS>"):
            sec = ln.strip()
        elif sec == "@<TRIPOS>ATOM" and ln.split():
            atom_idx.append(i)
    # The element comes from the atom NAME (C1, Br1, H10), not the type column:
    # antechamber writes GAFF2 types there (c3, n7, s6, cy), not SYBYL types
    # (C.3, N.ar), so splitting on "." yields "C3"/"N7" and never matches.
    t_elems = []
    for i in atom_idx:
        f = lines[i].split()
        t_elems.append("".join(c for c in f[1] if c.isalpha()).upper())
    p_elems = [a.GetSymbol().upper() for a in pose.GetAtoms()]
    if len(t_elems) != len(p_elems):
        raise ValueError(f"atom count mismatch: template {len(t_elems)} vs pose {len(p_elems)}")

    if t_elems == p_elems:
        mapping = list(range(len(p_elems)))             # same construction path
    else:
        tmpl = Chem.MolFromMol2File(str(template_mol2), removeHs=False, sanitize=False)
        if tmpl is None:
            raise ValueError("could not parse template mol2 for atom mapping")
        match = tmpl.GetSubstructMatch(pose)            # tmpl idx per pose atom
        if not match:
            raise ValueError("no substructure match between pose and template")
        mapping = [0] * len(match)
        for pose_i, tmpl_i in enumerate(match):
            mapping[tmpl_i] = pose_i

    conf = pose.GetConformer()
    for k, i in enumerate(atom_idx):
        f = lines[i].split()
        p = conf.GetAtomPosition(mapping[k])
        lines[i] = (f"{int(f[0]):>7} {f[1]:<8}{p.x:10.4f}{p.y:10.4f}{p.z:10.4f} "
                    f"{f[5]:<8}{int(f[6]):>4} {f[7]:<8}{float(f[8]):>10.6f}")
    out_mol2.write_text("\n".join(lines) + "\n")
    return len(atom_idx)


def run(cmd: list[str], cwd: Path, log: str) -> tuple[bool, str]:
    r = sp.run(cmd, cwd=str(cwd), stdout=sp.PIPE, stderr=sp.STDOUT, text=True, timeout=7200)
    (cwd / log).write_text(r.stdout)
    return r.returncode == 0, r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pose_pdb")
    ap.add_argument("out_dir")
    ap.add_argument("ligparam_dir", help="dir with lig.gaff2.mol2 and lig.frcmod for this ligand")
    ap.add_argument("--theory", default="PM6")
    ap.add_argument("--maxcyc", type=int, default=500)
    ap.add_argument("--qmcharge", type=int, default=0)
    ap.add_argument("--mol-name", dest="mol_name", default=None)
    args = ap.parse_args()

    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    nres, nlig = split_pose(Path(args.pose_pdb), d)
    print(f"protein residues {nres} | ligand atoms {nlig}")

    # protonate / clean the protein for Amber
    ok, out = run(["pdb4amber", "-i", "prot_raw.pdb", "-o", "prot.pdb", "--reduce", "--dry"],
                  d, "pdb4amber.log")
    if not (d / "prot.pdb").exists():
        ok, out = run(["pdb4amber", "-i", "prot_raw.pdb", "-o", "prot.pdb"], d, "pdb4amber2.log")
    print("pdb4amber:", "ok" if (d / "prot.pdb").exists() else "FAILED")

    lig_mol2 = Path(args.ligparam_dir) / "lig.gaff2.mol2"
    lig_frcmod = Path(args.ligparam_dir) / "lig.frcmod"
    for f in (lig_mol2, lig_frcmod):
        shutil.copy(f, d / f.name)
    shutil.copy(HEME_PREP, d / "HEM.in")
    shutil.copy(HEME_PRM, d / "HEM.prm")

    # Force the ligand residue name to LIG so the masks below are unambiguous.
    # Only the subst_name column inside @<TRIPOS>ATOM may be touched: a blanket
    # string replace rewrites "@<TRIPOS>MOLECULE" into "@<TRIPOS>LIGECULE" and
    # tleap then rejects the file as invalid MOL2.
    lines, section = [], ""
    for ln in (d / "lig.gaff2.mol2").read_text().splitlines():
        if ln.startswith("@<TRIPOS>"):
            section = ln.strip()
        elif section == "@<TRIPOS>ATOM":
            f = ln.split()
            if len(f) >= 8:
                ln = ln.replace(f"{f[7]}", "LIG", 1) if f[7] != "LIG" else ln
        lines.append(ln)
    (d / "lig.gaff2.mol2").write_text("\n".join(lines) + "\n")

    smiles = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    mid = args.mol_name or Path(args.ligparam_dir).name
    nat = retemplate_ligand(Path(args.pose_pdb), d / "lig.gaff2.mol2", smiles[mid], d / "lig_pose.mol2")
    print(f"ligand retemplated onto GAFF2 parameters: {nat} atoms")

    (d / "build.leap").write_text(f"""
source leaprc.protein.ff14SB
source leaprc.gaff2
loadamberprep HEM.in
loadamberparams HEM.prm
loadamberparams lig.frcmod
LIG = loadmol2 lig_pose.mol2
prot = loadpdb prot.pdb
heme = loadpdb heme.pdb
lig  = LIG
cplx = combine {{prot heme lig}}
saveamberparm cplx cplx.prmtop cplx.inpcrd
savepdb cplx cplx.pdb
quit
""")
    ok, out = run(["tleap", "-f", "build.leap"], d, "tleap.log")
    built = (d / "cplx.prmtop").exists()
    print("tleap:", "BUILT" if built else "FAILED")
    if not built:
        for ln in out.splitlines():
            if any(k in ln for k in ("FATAL", "Error", "could not", "Unknown")):
                print("   ", ln.strip()[:160])
        return

    # QM region: ligand + heme + the axial cysteine side chain
    qmmask = f"@%* & (:LIG | :HEM | (:{CYS_RESNUM} & @SG,CB))"
    (d / "min.in").write_text(f"""QM/MM minimisation, QM = ligand + heme + Cys{CYS_RESNUM} SG/CB
 &cntrl
  imin=1, maxcyc={args.maxcyc}, ncyc={args.maxcyc // 2}, ntmin=1,
  ntb=0, igb=1, cut=12.0,
  ntr=1, restraintmask='!(:MOL | :LIG | :HEM) & @CA,C,N,O', restraint_wt=10.0,
  ifqnt=1,
 /
 &qmmm
  qmmask=':MOL | :LIG | :HEM',
  qmcharge={args.qmcharge},
  qm_theory='{args.theory}',
  qmshake=0, qm_ewald=0, qmcut=12.0,
  writepdb=1, verbosity=0,
 /
""")
    # tleap writes an OLD-STYLE prmtop with no %FLAG ATOMIC_NUMBER, so sander
    # guesses the heme iron's element. Worse, the heme prep file's mass of
    # 55.0 is closer to Mn (54.938) than to Fe (55.845), so inferring from
    # mass silently yields MANGANESE - a run that converges cleanly and is
    # chemically meaningless. Set the element explicitly BY ATOM NAME and
    # re-save in modern format so sander reads 26.
    import parmed as pmd
    top = pmd.load_file(str(d / "cplx.prmtop"))
    nfix = 0
    for a in top.atoms:
        if a.residue.name == "HEM" and a.name.upper().startswith("FE"):
            if a.atomic_number != 26:
                a.atomic_number = 26
                nfix += 1
    top.save(str(d / "cplx_fix.prmtop"), overwrite=True)
    print(f"prmtop: set atomic_number=26 on {nfix} heme Fe atom(s)")
    bad = [(a.name, a.element_name) for a in pmd.load_file(str(d / "cplx_fix.prmtop")).atoms
           if a.residue.name == "HEM" and a.name.upper().startswith("FE")]
    print("heme metal now reads as:", bad)

    ok, out = run(["sander", "-O", "-i", "min.in", "-p", "cplx_fix.prmtop", "-c", "cplx.inpcrd",
                   "-o", "min.out", "-r", "min.rst", "-ref", "cplx.inpcrd"], d, "sander.log")
    done = (d / "min.rst").exists()
    print(f"sander QM/MM ({args.theory}):", "DONE" if done else "FAILED")
    if not done:
        tail = (d / "min.out").read_text()[-1500:] if (d / "min.out").exists() else out[-1500:]
        print(tail)


if __name__ == "__main__":
    main()
