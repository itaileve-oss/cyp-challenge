# OrbMol machine-learned potential

Energies and forces for the heme site from an ωB97M-trained machine-learned
interatomic potential (MLIP), as a replacement for semiempirical QM (PM6, GFN2-xTB).
Post-deadline tooling: none of this was used in a submission.

Code: `structure/src/orb_calculator.py` (ASE calculator, interaction energy),
`structure/src/orb_systems.py` (protonated heme-site builder),
`structure/src/validate_orbmol.py` (the checks below).

## Model choice

| | `orbmol_v2` | `orb_v3_conservative_omol` |
|---|---|---|
| long-range electrostatics | latent partial charges + `coulomb_module` | none (6 Å cutoff, charge as embedding) |
| conservative forces | yes | yes |
| 66 atoms, CPU | 11.1 s | 4.6 s |
| 66 atoms, GPU (T500) | 0.80 s | 0.61 s |

On a GPU the electrostatics cost only 1.3×, so **`orbmol_v2` is the default**.
Both models condition on total charge and spin multiplicity through
`atoms.info["charge"]` / `atoms.info["spin"]`, so these must be set correctly.

## Environment (conda `mlip`)

`orb-models 0.7.0` removed the `ORBCalculator` that older examples use: the
loader returns `(model, adapter)`, and prediction is
`adapter.from_ase_atoms(atoms)` → `model.predict(batch)` with forces in
`grad_forces`. `OrbCalculator` wraps this pair as a standard ASE calculator.

GPU pins for a CUDA 12.x driver:
- `torch==2.14.1+cu126`: the default `+cu132` build needs driver ≥ 13.
- `warp-lang==1.15.0`: NVIDIA Warp builds the neighbour list, and 1.17+ ships
  CUDA 13 only. It silently hangs on a 12.x driver.

The first call compiles Warp kernels (about 100–160 s), which are then cached
in `~/.cache/warp`.

## Hydrogens are not optional

Every pose in this project is heavy-atom only. OMol25 contains complete
closed-valence molecules, so a heavy-atom heme is an undersaturated radical
the model has never seen. Early benchmarks on such systems gave max forces of
10 eV/Å and a −88 kcal/mol "interaction energy". Those were numbers, not
chemistry.

`orb_systems.heme_site(pose_pdb, lig_h.sdf)` builds the protonated site:
- **Heme.** The 30 C–H come from the CCD `HEM` entry (`structure/params/ccd/HEM.cif`),
  whose atom names match the Boltz/GalaxyDock heme. Each H is placed by
  Kabsch-superposing the CCD ideal geometry of its parent atom and the parent's
  neighbours onto the pose, with no valence perception near Fe. CCD H use X-ray
  C–H lengths (0.91–0.98 Å) and are rescaled to 1.08/1.09 Å. The acid H
  (H2A, H2D) are dropped, giving carboxylates.
- **Ligand.** Bond orders come from `lig_h.sdf` (a different conformer, not the
  pose), are assigned onto the pose heavy atoms, and RDKit then adds H.
- **Cys442.** Truncated to CB–SG and capped as methylthiolate.
- **Charge.** Fe(III) site = −2, plus the ligand's formal charge. Tags:
  1 = ligand, 0 = host.

## Validation: OCNT-2395436, GalaxyDock2-HEME pose, N-coordinated

122 atoms (54 H), charge −2, T500 GPU.

| check | result |
|---|---|
| geometry | min H–H 1.66 Å, min H–heavy 1.03 Å (N–H), no clashes |
| finite-difference forces | max error 0.004 eV/Å (max force 2.7 eV/Å) → conservative, MD-safe |
| speed | 1.3 s/call (122 atoms); first call 160 s (kernel compile) |
| Fe(III) spin states | doublet 0, quartet +5.8, sextet +15.1 kcal/mol |
| ΔE_int at pose (doublet) | −6.2 kcal/mol |
| ligand relaxed, heme site frozen | Fe–N 2.46 → **2.19 Å** |

What the results mean:
- **Spin.** The low-spin doublet is lowest, which is correct for six-coordinate
  Fe(III) with an N donor trans to thiolate (type-II spectra). Earlier runs
  used multiplicity 6, the wrong state for an N-coordinated pose.
- **Fe–N.** The crystal range is 2.20–2.35 Å. sander/PM6 gave 1.96 Å on this
  pose. OrbMol lands at the edge of the crystal range with no restraint and no
  invented metal parameters.
- **Caveats.**
  - The relaxation did not reach fmax 0.05 in 300 FIRE steps.
  - The −44 kcal/mol relaxation energy is mostly ligand internal strain in vacuum,
    with no protein to hold the ligand.
  - The ΔE_int keeps the host in the doublet. A five-coordinate thiolate heme
    alone is high-spin, so a spin-adiabatic ΔE_int would differ.

Re-run:

```bash
conda activate mlip
python structure/src/validate_orbmol.py \
  structure/analysis/seed1/qmmm_inputs/GD2_OCNT-2395436.pdb \
  structure/analysis/seed1/ligparams/OCNT-2395436/lig_h.sdf \
  --device cuda --out structure/analysis/orbmol/validate_2395436_v2.json
```

## Limits

- **No OpenMM-ML route yet.** `openmm-torch` needs a TorchScript module that
  takes positions and returns energy, graph build included. Orb builds its
  graph with Warp in Python, so `MLPotential.createMixedSystem()` does not work
  as is.
- **ML/MM today means ASE's subtractive scheme** (`SimpleQMMM` / `ForceQMMM`),
  which is mechanical embedding only. `orbmol_v2` has no external point-charge
  input, so the ML region does not feel the protein's field. Keeping whole
  first-shell residues inside the ML region partly compensates.
- Pocket-scale systems (about 500–600 atoms with H) are affordable because cost is
  near-linear in atom count. MLIP-driven MD on the T500 (about 1 s/step) is not.
