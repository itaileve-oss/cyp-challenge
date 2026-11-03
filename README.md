# OpenADMET CYP Blind Challenge — methods

Code for both tracks of the [OpenADMET CYP inhibition blind challenge](https://huggingface.co/spaces/openadmet/cyp-challenge):
pIC50 regression for CYP1A2/2C9/2D6/3A4, and CYP3A4–ligand complex prediction.

Everything here is reproducible from public inputs: the challenge data, UniProt
P08684, the heme CCD component, public MSAs, and deposited CYP3A4 structures. No
proprietary data.

Written up in full: [regression methods](https://claude.ai/artifact/JjKfzrhtLgSuCED9AUD6KA) ·
[structure methods](https://claude.ai/artifact/HKqBErguaofNXPuKFE7sDU)

---

## Structure track — `structure/src/`

### Pose generation

| script | what it does |
|---|---|
| `make_boltz_inputs.py` | writes one Boltz-2 YAML per ligand: protein chain + `ccd: HEM` cofactor chain + ligand from the verbatim challenge SMILES. Optional Cys442–Fe bond constraint. |
| `calibrate_boltz.sh` | measures wall-clock and peak VRAM per sampling config, so the real per-sample cost is known before committing to a pool size |
| `overnight.sh` | unattended pipeline: waits for generation, shards analysis per ligand across cores, scores, clusters, selects, builds and validates submission zips |
| `pull_results.sh` | detached local watcher that downloads results from a remote instance when a marker file appears |

Calibration result worth knowing: plain Boltz-2 defaults beat every "informed"
adjustment. `--use_potentials` destroys heme coordination outright (100% → 0% of
poses coordinating), an explicit Cys442–Fe bond constraint is mildly harmful, and
lowering `step_scale` trades a collapse in PoseBusters validity (100% → 22%) for
almost no extra diversity.

### Docking

| script | engine | notes |
|---|---|---|
| `dock_gd2heme_structure.py` | **GalaxyDock2-HEME** | the right engine for a P450 — explicit heme term. Requires UCSF Chimera on PATH; heme must be resname `HEM` as HETATM **in the protein's own chain**; GD2 parses fixed-length path fields, so run from a short absolute path |
| `dock_pool.py` | **GNINA / Vina** | kept as a negative control only. Vina-type scoring cannot see an Fe–N dative bond: 0/20 coordinating modes for a ligand Boltz coordinates in 20/20 |

Ligands are embedded from the verbatim challenge SMILES with no protonation
change — an imidazole protonated at pH 7.4 loses the lone pair it needs to reach
the iron.

### Scoring and analysis

| script | what it measures |
|---|---|
| `pb_official.py` | the organizers' exact PoseBusters path, including their tolerant bond-order matcher. `POSEBUSTERS_MAX_FAILURES = 0`, so one failed check zeroes LDDT-PLI |
| `analyze_pool.py` | per-pose geometry: Fe distance and closest-atom element, distal-vs-proximal heme face, pocket contacts, Boltz confidence fields |
| `protein_variability.py` | receptor variability, raw and after Kabsch fit — the script that found the frame problem below |
| `cluster_poses.py` | frame-corrected clustering: CA superposition, then symmetry-aware in-place RMSD, then Butina at 2 Å |
| `motif_score.py`, `motif_score_gd2.py` | agreement with a contact motif derived from deposited CYP3A4 complexes |
| `analyze_gd2heme.py` | cross-method comparison of docked and co-folded poses in a shared frame |
| `xtb_score.py` | GFN2-xTB interaction energy on a truncated active site (ligand + heme + Cys442) |
| `qmmm_minimize.py` | QM/MM minimisation with sander: QM region = ligand + heme, MM = ff14SB protein |
| `select_poses.py`, `select_final.py` | candidate selections under different criteria |
| `build_submission.py` | assembles the zip and runs the organizers' validator |
| `param_ligands.py` | GAFF2/AM1-BCC parameters per ligand for MM-PBSA or MM/QM work |

### Four findings that cost time to discover

**Boltz emits every diffusion sample in its own rigid-body frame.** Raw CA RMSD
between two samples of one ligand is 34.5 Å; after superposition it is 1.40 Å.
Comparing ligand coordinates without fitting first measures the frame, not the
pose, and inflates apparent diversity from ~2 Å to ~8 Å.

**The predicted receptor is rigid.** Backbone RMSD is 1.40 Å between samples of
one ligand and 1.44 Å between *different* ligands; pocket side chains 0.64 vs
0.78 Å. One pocket conformation regardless of what is bound — so no induced fit,
and a ceiling of ~12 contacted residues against a crystal median of 16.

**Distance is not a test of coordination.** GalaxyDock reproduces crystal Fe–N
distances well but approaches ~49° off the nitrogen lone-pair axis; co-folding
sits slightly short but points the lone pair at the metal. Measure the angle.

**Consensus selection does not transfer.** Boltz confidence is nearly flat across
samples (0.946–0.990) while samples sit several Å apart, so confidence cannot
rank them — and picking the modal cluster or maximising a burial score
concentrates the model's orientation bias. Our motif/medoid selection finished
*below* the trivial baseline of taking the highest-confidence valid sample.

### Force-field notes for metalloprotein work

A fixed-charge force field cannot hold an Fe–N dative bond. With the Li/Merz
Fe³⁺ 12-6 parameters the Fe–N minimum sits at 3.21 Å, so minimisation pushes the
nitrogen *away* from the crystal distance; CHARMM36 fails the other way, with
ε<sub>Fe</sub> = 0 and q = +0.24 giving no repulsive wall and monotonic
attraction. Hence QM/MM. Heme MM parameters in `structure/params/heme/` — use the
pentacoordinated set, whose heavy-atom names match Boltz's heme and which has no
axial water. Watch the prep file's `MASS FE 55.0`: that is closer to manganese
(54.938) than iron (55.845), so element inference from mass silently yields Mn,
and the run converges cleanly while being chemically meaningless.

---

## Regression track — `src/`

Multitask ensemble over all four isoforms, per-isoform structure and
pharmacophore features blended on top, then an affine placement correction.

| area | scripts |
|---|---|
| features | `features_*.py` — descriptors, Uni-Mol embeddings, pharmacophores, logD |
| docking features | `prepare_cyp3a4_receptor.py`, `prepare_ligands_gd2heme*.py`, `run_batch_ensemble.py`, `launch_*_docking*.sh`, `parse_docking_results.py` |
| models | `cv_*.py` — cross-validation per isoform and feature set |
| ensembling | `build_final_ensemble.py`, `sweep_standalone_blend_weights.py`, `tabpfn_confirm_and_predict.py` |
| calibration | `apply_placement_calibration.py`, `check_cyp3a4_rho.py` |
| submission | `build_test_submission.py`, `build_winning_poses_general.py` |

Two notes. TabPFN is *worse* than the ensemble alone at every endpoint and was
adopted only as a blend component — its value is decorrelation (out-of-fold
correlation with the LightGBM member is 0.58 for CYP2D6), not accuracy. And the
placement step corrects spread, never location: out of fold the optimal additive
offset is zero at every endpoint by construction, because cross-validation holds
the train-to-test gap constant. Finding a location offset needs leaderboard
feedback.

---

## Environments

Four conda environments, by role:

- **`oadmet_cyp_tutorial`** — the challenge's own spec; RDKit, LightGBM, datasets
- **`GNINA`** — GNINA plus RDKit, gemmi, PoseBusters, MDAnalysis, OpenMM. Most analysis runs here
- **`openfe`** — full AmberTools (tleap, antechamber, parmchk2, sander, MMPBSA.py), parmed, openmmforcefields
- **`xtb_env`** — GFN2-xTB. Set `ulimit -s unlimited` and `OMP_STACKSIZE=4G` or xtb segfaults on large systems, and run single-threaded: 8 threads on a 70-atom system is pure overhead

External tools not vendored here: Boltz-2, GalaxyDock2-HEME (needs UCSF Chimera),
and the challenge tutorial repo.

## Layout

```
src/                 regression track
structure/src/       structure track
structure/params/    heme force-field parameters, crystal motif reference
outputs/versions/    submission history with a changelog
notebooks/           EDA
```

Generated pose pools, trained models and challenge data are not tracked — see
`.gitignore`. Remote-execution scripts take `REMOTE_HOST` and `REMOTE_PORT` from
the environment.
