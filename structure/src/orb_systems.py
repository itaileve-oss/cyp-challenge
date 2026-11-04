"""Build fully protonated heme-site systems for OrbMol from heavy-atom poses.

Every pose in this project (Boltz-2, GalaxyDock2-HEME, the QM/MM inputs) is
heavy-atom only. OrbMol is trained on OMol25, which is made of complete,
closed-valence molecules, so a heavy-atom heme is an undersaturated radical the
model has never seen. Energies on such a system are numbers, not chemistry.
This module adds the hydrogens back:

* **Heme** -- the 30 C-H hydrogens come from the PDB Chemical Component
  Dictionary entry HEM (``structure/params/ccd/HEM.cif``), whose atom names
  match the Boltz/GalaxyDock heme exactly. Each H is placed by superposing the
  CCD ideal geometry of its parent atom and the parent's heavy neighbours onto
  the pose, so only local geometry is used and no valence perception is needed
  around the iron. The two carboxylic-acid hydrogens (H2A, H2D) are dropped:
  propionates are carboxylates at physiological pH.
* **Ligand** -- bond orders come from the parametrised ``lig_h.sdf`` (which is
  a different conformer, not the pose), assigned onto the pose heavy atoms, then
  RDKit places hydrogens on the pose geometry.
* **Cys442** -- truncated to CB-SG and capped as methylthiolate (CH3-S-), the
  standard minimal model of the proximal ligand.

Charge bookkeeping (Fe(III)): porphyrin(2-) + 2 carboxylates(2-) + Fe(3+) +
thiolate(1-) = -2 for the heme site, plus the ligand's formal charge.
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
HEM_CCD = ROOT / "structure/params/ccd/HEM.cif"
CYS_RESNUM = 442
ACID_H = {"H2A", "H2D"}


def _kabsch(P, Q):
    """Rotation R and translation t minimising |R P + t - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def _ccd_heme():
    doc = gemmi.cif.read(str(HEM_CCD))
    b = doc.sole_block()
    atoms = b.find("_chem_comp_atom.", ["atom_id", "type_symbol",
                                        "pdbx_model_Cartn_x_ideal",
                                        "pdbx_model_Cartn_y_ideal",
                                        "pdbx_model_Cartn_z_ideal"])
    ideal, elem = {}, {}
    for r in atoms:
        name = r.str(0)
        ideal[name] = np.array([float(r[2]), float(r[3]), float(r[4])])
        elem[name] = r.str(1)
    nbrs = {n: [] for n in ideal}
    for r in b.find("_chem_comp_bond.", ["atom_id_1", "atom_id_2"]):
        a1, a2 = r.str(0), r.str(1)
        nbrs[a1].append(a2)
        nbrs[a2].append(a1)
    return ideal, elem, nbrs


def protonate_heme(heme_res: gemmi.Residue):
    """Return [(symbol, xyz)] for heme heavy atoms plus placed hydrogens."""
    ideal, elem, nbrs = _ccd_heme()
    pose = {a.name: np.array(a.pos.tolist()) for a in heme_res if a.element.name != "H"}
    out = [(a.element.name.capitalize() if len(a.element.name) > 1 else a.element.name,
            pose[a.name]) for a in heme_res if a.name in pose]
    for h, e in elem.items():
        if e != "H" or h in ACID_H:
            continue
        parent = nbrs[h][0]
        # parent + its heavy neighbours (+ their neighbours if too few for a frame)
        frame = [parent] + [n for n in nbrs[parent] if elem[n] != "H"]
        if len(frame) < 3:
            frame += [m for n in frame[1:] for m in nbrs[n]
                      if elem[m] != "H" and m not in frame]
        frame = [n for n in frame if n in pose][:6]
        R, t = _kabsch(np.array([ideal[n] for n in frame]),
                       np.array([pose[n] for n in frame]))
        # CCD ideal H use X-ray-style C-H (0.91-0.98 A); keep the direction,
        # reset the length to the neutron/QM value the model was trained on
        v = R @ ideal[h] + t - pose[parent]
        length = 1.08 if len(nbrs[parent]) <= 3 else 1.09     # sp2 / sp3
        out.append(("H", pose[parent] + length * v / np.linalg.norm(v)))
    return out


def protonate_ligand(lig_res: gemmi.Residue, template_sdf):
    """Pose-geometry ligand with hydrogens, bond orders taken from template."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    tmpl = Chem.MolFromMolFile(str(template_sdf), removeHs=False)
    tmpl_noH = Chem.RemoveHs(tmpl)
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(lig_res))
    heavy = [a for a in lig_res if a.element.name != "H"]
    for i, a in enumerate(heavy):
        rw.AddAtom(Chem.Atom(a.element.atomic_number))
        conf.SetAtomPosition(i, a.pos.tolist())
    # connectivity from distances (pose geometry), orders from template
    xyz = np.array([a.pos.tolist() for a in heavy])
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    cov = np.array([gemmi.Element(a.element.name).covalent_r for a in heavy])
    for i in range(len(heavy)):
        for j in range(i + 1, len(heavy)):
            if d[i, j] < 1.25 * (cov[i] + cov[j]):
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.AddConformer(conf, assignId=True)
    m = AllChem.AssignBondOrdersFromTemplate(tmpl_noH, m)
    charge = Chem.GetFormalCharge(m)
    mh = Chem.AddHs(m, addCoords=True)
    pos = mh.GetConformer().GetPositions()
    return [(a.GetSymbol(), pos[i]) for i, a in enumerate(mh.GetAtoms())], charge


def cys_cap(cys_res: gemmi.Residue):
    """Methylthiolate from Cys CB/SG; three H on CB, one along the CB-CA bond."""
    p = {a.name: np.array(a.pos.tolist()) for a in cys_res}
    cb, sg, ca = p["CB"], p["SG"], p["CA"]
    u = (ca - cb) / np.linalg.norm(ca - cb)          # becomes one C-H direction
    w = (sg - cb) / np.linalg.norm(sg - cb)
    # tetrahedral set around CB: rotate u about the CB-SG axis by 120/240 deg
    hs = []
    for ang in (0, 120, 240):
        th = np.radians(ang)
        v = (u * np.cos(th) + np.cross(w, u) * np.sin(th)
             + w * np.dot(w, u) * (1 - np.cos(th)))
        hs.append(cb + 1.09 * v / np.linalg.norm(v))
    return [("C", cb), ("S", sg)] + [("H", h) for h in hs]


def heme_site(pose_pdb, template_sdf, ligand="LIG"):
    """Protonated ligand + heme + methylthiolate as tagged ``ase.Atoms``.

    Tags: 1 = ligand, 0 = heme site. ``atoms.info`` carries the total charge
    and the charges of each fragment so ``interaction_energy`` can be called
    with the right values.
    """
    st = gemmi.read_structure(str(pose_pdb))
    st.remove_hydrogens()
    res = {r.name if r.name != "CYS" else ("CYS" if r.seqid.num == CYS_RESNUM else "x"): r
           for ch in st[0] for r in ch
           if r.name in (ligand, "HEM") or (r.name == "CYS" and r.seqid.num == CYS_RESNUM)}
    lig, qlig = protonate_ligand(res[ligand], template_sdf)
    host = protonate_heme(res["HEM"]) + cys_cap(res["CYS"])
    syms = [s for s, _ in lig + host]
    xyz = np.array([x for _, x in lig + host])
    at = Atoms(symbols=syms, positions=xyz)
    at.set_tags([1] * len(lig) + [0] * len(host))
    at.info.update(charge=qlig - 2, charge_ligand=qlig, charge_host=-2)
    return at
