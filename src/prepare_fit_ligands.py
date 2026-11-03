"""Fit-molecule preparation for structure-based pose-alignment features.

Terminology (ROCS convention): "query" = a molecule with a known 3D pose we align
onto (here, real PDB-bound CYP inhibitor structures, see fetch_cyp_queries.py /
prepare_query_ligands.py); "fit" = a molecule we're searching for a pose for by
superposing it onto a query (here, our own training/test SMILES, no experimental
structure).

For each of our own SMILES, produce a protonated, multi-conformer 3D representation
ready for O3A alignment onto a query's bound pose:
  1. Protonate at pH 7.4 via dimorphite_dl -- take the top-ranked microstate rather
     than enumerating every combinatorial variant (dimorphite's default pH *window*
     returns all per-site uncertainty combinations independently, which explodes
     combinatorially and includes some chemically implausible states for sites like
     sulfonamide nitrogens; a single pH point is a more defensible default state).
  2. Embed a CONFORMER ENSEMBLE (not one arbitrary geometry, unlike every earlier
     xTB-descriptor experiment in this project) via ETKDGv3 with RMS pruning, MMFF-
     optimized. The alignment step downstream picks whichever conformer superposes
     best onto the chosen query -- more conformers give it a better chance of
     finding something close to the true bioactive geometry.

Run in the `GNINA` conda env (has dimorphite_dl; oadmet_cyp_tutorial does not).
"""

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dimorphite_dl
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

DEFAULT_PH = 7.4
DEFAULT_NUM_CONFS = 20
DEFAULT_PRUNE_RMS = 0.5
SEED = 42


def protonate_dominant(smiles: str, ph: float = DEFAULT_PH) -> str | None:
    """Return the top-ranked protonated SMILES at a single pH point (not a window),
    to avoid dimorphite's combinatorial per-site uncertainty enumeration. None on
    failure.
    """
    try:
        variants = dimorphite_dl.protonate_smiles(smiles, ph_min=ph, ph_max=ph)
    except Exception:
        return None
    return variants[0] if variants else None


def embed_conformer_ensemble(
    smiles: str,
    num_confs: int = DEFAULT_NUM_CONFS,
    prune_rms_thresh: float = DEFAULT_PRUNE_RMS,
    seed: int = SEED,
) -> Chem.Mol | None:
    """Embed a pruned conformer ensemble (explicit Hs, MMFF-optimized) for one
    SMILES. Returns a single multi-conformer RDKit Mol, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms_thresh
    params.useRandomCoords = True
    try:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if len(conf_ids) == 0:
            return None
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        # Drop conformers that failed to converge/optimize (result code != 0), unless
        # that would drop everything.
        converged = [cid for cid, (code, _) in zip(conf_ids, results) if code == 0]
        if converged:
            for cid in list(conf_ids):
                if cid not in converged:
                    mol.RemoveConformer(cid)
    except Exception:
        return None
    return mol if mol.GetNumConformers() > 0 else None


def prepare_fit_molecule(
    smiles: str,
    ph: float = DEFAULT_PH,
    num_confs: int = DEFAULT_NUM_CONFS,
) -> Chem.Mol | None:
    """Full prep: protonate (dominant microstate) then embed a conformer ensemble.
    Falls back to the original (unprotonated) SMILES if protonation fails, since a
    neutral-form conformer ensemble is still better than nothing.
    """
    protonated = protonate_dominant(smiles, ph=ph)
    target_smiles = protonated if protonated is not None else smiles
    mol = embed_conformer_ensemble(target_smiles, num_confs=num_confs)
    if mol is None and protonated is not None:
        # Protonated form failed to embed (can happen for unusual charged geometries)
        # -- fall back to the neutral form rather than losing the molecule entirely.
        mol = embed_conformer_ensemble(smiles, num_confs=num_confs)
    return mol


def _run_one(args: tuple[int, str]) -> tuple[int, Chem.Mol | None]:
    idx, smiles = args
    return idx, prepare_fit_molecule(smiles)


def prepare_fit_molecules_batch(smiles_list: list[str], n_workers: int = 8) -> list[Chem.Mol | None]:
    results: list[Chem.Mol | None] = [None] * len(smiles_list)
    tasks = list(enumerate(smiles_list))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for idx, mol in pool.map(_run_one, tasks):
            results[idx] = mol
    return results


if __name__ == "__main__":
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    data_dir = PROJECT_ROOT / "data" / "raw"
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    smiles = tr["SMILES"].tolist()

    print(f"Preparing {len(smiles)} fit molecules (protonation + conformer ensembles)...", flush=True)
    start = time.time()
    mols = prepare_fit_molecules_batch(smiles, n_workers=8)
    elapsed = time.time() - start
    n_ok = sum(m is not None for m in mols)
    n_confs = [m.GetNumConformers() for m in mols if m is not None]
    print(f"Done in {elapsed:.1f}s. {n_ok}/{len(smiles)} succeeded.", flush=True)
    if n_confs:
        print(
            f"Conformers per molecule: min={min(n_confs)}, "
            f"median={sorted(n_confs)[len(n_confs)//2]}, max={max(n_confs)}"
        )

    out_path = PROJECT_ROOT / "models" / "fit_conformers_train_cache.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(mols, f)
    print(f"Saved to {out_path}")
