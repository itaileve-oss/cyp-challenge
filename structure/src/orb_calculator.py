"""ASE Calculator for Orb MLIPs, plus helpers for building systems from PDB.

orb-models 0.7.0 removed the `ORBCalculator` that older examples use: the
pretrained loaders now return a `(model, adapter)` tuple and prediction goes
`adapter.from_ase_atoms(atoms)` -> `model.predict(batch)`. This wraps that pair
in a standard ASE Calculator so ASE optimisers, molecular dynamics and anything
else expecting the Calculator interface work unchanged.

Model choice matters:

* `orbmol_v2` predicts latent partial charges and adds an explicit
  `coulomb_module`, so it carries LONG-RANGE electrostatics. Use it for
  interaction energies, binding free energies, and anything where a charged
  species or a solvent-exposed pocket matters.
* `orb_v3_conservative_omol` has no Coulomb term — short range only, with total
  charge entering as an embedding. Slightly cheaper; use it for speed
  comparisons, not for energetics that depend on electrostatics.

Both are *conservative* models, meaning forces are the analytic gradient of the
energy rather than a separate prediction head. That is required for molecular
dynamics and for metadynamics, where non-conservative forces silently break
energy conservation.

Charge and spin are read from `atoms.info["charge"]` and `atoms.info["spin"]`,
which is how the OMol25-trained models are conditioned. Setting them wrongly
changes the electronic state being modelled, so they are not optional: a heme
with two deprotonated propionates and Fe(III) is charge -1, and high-spin d5
gives spin multiplicity 6.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

_LOADERS = {
    "orbmol_v2": "orbmol_v2",
    "orb_v3_omol": "orb_v3_conservative_omol",
    "orb_v3_direct_omol": "orb_v3_direct_omol",
}


def load_orb(model: str = "orbmol_v2", device: str = "auto",
             precision: str = "float32-high"):
    """Load a pretrained Orb model and its atoms adapter."""
    from orb_models.forcefield import pretrained

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    name = _LOADERS.get(model, model)
    loader = getattr(pretrained, name)
    mdl, adapter = loader(device=device, precision=precision)
    return mdl, adapter, device


class OrbCalculator(Calculator):
    """ASE Calculator backed by an Orb forcefield model.

    Parameters
    ----------
    model_name:
        Key in ``_LOADERS`` or a loader name from ``orb_models.forcefield.pretrained``.
    device:
        ``"auto"``, ``"cuda"`` or ``"cpu"``.
    charge, spin:
        Defaults used when an ``Atoms`` object carries nothing in ``.info``.
        Spin here is the multiplicity (1 = closed shell, 6 = high-spin d5 Fe(III)).
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, model_name: str = "orbmol_v2", device: str = "auto",
                 charge: int = 0, spin: int = 1, precision: str = "float32-high",
                 **kwargs):
        super().__init__(**kwargs)
        self.model, self.adapter, self.device = load_orb(model_name, device, precision)
        self.model_name = model_name
        self.default_charge = charge
        self.default_spin = spin

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        at = self.atoms.copy()
        at.info.setdefault("charge", self.default_charge)
        at.info.setdefault("spin", self.default_spin)
        batch = self.adapter.from_ase_atoms(at, device=self.device)
        out = self.model.predict(batch)
        energy = float(out["energy"].detach().cpu())
        forces = out["grad_forces"].detach().cpu().numpy().astype(np.float64)
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


# --------------------------------------------------------------------------- #
# building systems from a complex PDB
# --------------------------------------------------------------------------- #

def atoms_from_pdb(path, keep_resnames=None, within=None, of_resname="LIG",
                   drop_hydrogens=True):
    """Build an ``ase.Atoms`` from a PDB.

    keep_resnames:
        Keep only these residue names (e.g. ``("LIG", "HEM")``).
    within:
        Alternatively, keep whole residues with any heavy atom within this many
        angstrom of ``of_resname``. Residues are kept whole, never cut through,
        because a truncated residue leaves dangling valences the model has never
        seen.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    if drop_hydrogens:
        st.remove_hydrogens()
    model = st[0]

    residues = [(ch.name, res) for ch in model for res in ch]
    if within is not None:
        ref = np.array([a.pos.tolist() for _, r in residues if r.name == of_resname
                        for a in r])
        if not len(ref):
            raise ValueError(f"no residue named {of_resname} in {path}")
        keep = []
        for cname, r in residues:
            xyz = np.array([a.pos.tolist() for a in r])
            if np.linalg.norm(xyz[:, None] - ref[None], axis=2).min() <= within:
                keep.append((cname, r))
        residues = keep
    elif keep_resnames is not None:
        residues = [(c, r) for c, r in residues if r.name in set(keep_resnames)]

    symbols, positions, tags = [], [], []
    for _, r in residues:
        for a in r:
            symbols.append(a.element.name)
            positions.append(a.pos.tolist())
            tags.append(1 if r.name == of_resname else 0)   # 1 = ligand
    from ase import Atoms
    at = Atoms(symbols=symbols, positions=np.array(positions))
    at.set_tags(tags)
    return at


def interaction_energy(calc, complex_atoms, ligand_tag=1,
                       charge_complex=0, charge_ligand=0, charge_host=0,
                       spin_complex=1, spin_ligand=1, spin_host=1):
    """Vertical interaction energy at the complex geometry.

    ``dE = E(complex) - E(host) - E(ligand)``, every term evaluated at the pose
    geometry with no relaxation, so the result is an interaction energy and not
    a binding energy: it excludes strain and desolvation. Comparable only
    between poses of the same ligand in the same host, which is what pose
    ranking needs.
    """
    tags = complex_atoms.get_tags()
    lig = complex_atoms[tags == ligand_tag]
    host = complex_atoms[tags != ligand_tag]

    def energy(at, charge, spin):
        at = at.copy()
        at.info["charge"] = charge
        at.info["spin"] = spin
        at.calc = calc
        return at.get_potential_energy()

    e_cx = energy(complex_atoms, charge_complex, spin_complex)
    e_host = energy(host, charge_host, spin_host)
    e_lig = energy(lig, charge_ligand, spin_ligand)
    return dict(e_complex=e_cx, e_host=e_host, e_ligand=e_lig,
                dE_eV=e_cx - e_host - e_lig,
                dE_kcal=(e_cx - e_host - e_lig) * 23.060548)
