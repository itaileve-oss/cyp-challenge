"""features_pharmacophore_other_isoforms.py was built earlier this session but
never actually validated -- it isn't imported by any pipeline. It encodes each
non-CYP3A4 isoform's own well-established SAR directly from 2D/3D ligand
structure (no docking dependency, so testable now while the CYP1A2/2C9/2D6
docking run is still in flight):

- CYP2D6: basic amine presence + its distance to the nearest aromatic ring
  centroid (literature: ionic/H-bond interaction with Glu216/Asp301, ~5-7A
  from the oxidation site)
- CYP2C9: acidic-group presence/count (salt bridge with Arg108)
- CYP1A2: planarity (CalcPBF) + aromatic fraction/ring count (small flat
  active site)

Same repeated randomized-cluster-fold harness and tuned LightGBM as the
CYP3A4 work, run independently per isoform against that isoform's own
pIC50_direct_inhibition endpoint and its own compound set.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import ttest_rel
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import useful_rdkit_utils as uru  # noqa: E402
from evaluation.config import ACTIVITY_METRICS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from features_pharmacophore_other_isoforms import (  # noqa: E402
    cyp1a2_feature_matrix,
    cyp2c9_feature_matrix,
    cyp2d6_feature_matrix,
)

N_FOLDS = 5
N_REPEATS = 8
TUNED = dict(num_leaves=15, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0,
             colsample_bytree=0.7, n_estimators=300, learning_rate=0.03)
METRIC_NAMES = [name for name, _ in ACTIVITY_METRICS]

ISOFORMS = {
    "CYP2D6": {"needs_conf": True, "builder": cyp2d6_feature_matrix},
    "CYP2C9": {"needs_conf": False, "builder": cyp2c9_feature_matrix},
    "CYP1A2": {"needs_conf": True, "builder": cyp1a2_feature_matrix},
}


def _mol_with_conf(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    cid = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if cid < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    except Exception:
        pass
    return mol


def build_mols(smiles_list: list[str], needs_conf: bool) -> list:
    if not needs_conf:
        return [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = []
    n_failed = 0
    for i, smi in enumerate(smiles_list):
        m = _mol_with_conf(smi)
        if m is None:
            n_failed += 1
        mols.append(m)
        if (i + 1) % 300 == 0:
            print(f"    conformers {i+1}/{len(smiles_list)} (failed={n_failed})", flush=True)
    print(f"    conformers done: {len(smiles_list)-n_failed}/{len(smiles_list)} succeeded", flush=True)
    return mols


def all_metrics(oof: np.ndarray, tr: pd.DataFrame, endpoint: str) -> dict:
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], endpoint: oof})
    ground_truth = tr[["Molecule_Name", endpoint, f"{endpoint}_conf_low", f"{endpoint}_conf_high"]]
    boot = score_activity_predictions(predictions, ground_truth, [endpoint])
    track = add_macro_endpoint(boot, [endpoint], ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return {m: float(summary.loc[endpoint, f"{m}_mean"]) for m in METRIC_NAMES}


def run_fold_split(X: np.ndarray, y: np.ndarray, cluster_labels: np.ndarray, params=None) -> np.ndarray:
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = np.full(len(y), np.nan)
    for tr_idx, te_idx in gkf.split(X, y, groups=cluster_labels):
        kwargs = dict(random_state=42, verbosity=-1)
        if params:
            kwargs.update(params)
        model = LGBMRegressor(**kwargs)
        model.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = model.predict(X[te_idx])
    return oof


def run_isoform(isoform: str, spec: dict, tr_all: pd.DataFrame) -> None:
    endpoint = f"{isoform}_pIC50_direct_inhibition"
    tr = tr_all[tr_all[endpoint].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[endpoint].to_numpy(dtype=float)
    print(f"\n=== {isoform}: {len(tr)} compounds ===", flush=True)

    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    X_base = np.hstack([fp, desc])

    print(f"  Building {isoform}-specific pharmacophore features...", flush=True)
    mols = build_mols(smiles, spec["needs_conf"])
    pharma = spec["builder"](mols)
    X_pharma = np.hstack([X_base, pharma])

    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    labels = ["baseline", f"+{isoform} pharma"]
    feature_sets = [X_base, X_pharma]
    all_results = {lab: {m: [] for m in METRIC_NAMES} for lab in labels}
    st_rae_results = []

    rng = np.random.default_rng(123)
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(unique_clusters))
        relabeled = perm[cluster_int]
        row_strae = []
        for lab, X in zip(labels, feature_sets):
            m = all_metrics(run_fold_split(X, y, relabeled, TUNED), tr, endpoint)
            for k, v in m.items():
                all_results[lab][k].append(v)
            row_strae.append(m["ST-RAE"])
        print(f"  repeat {rep}: " + "  ".join(f"{lab}={v:.4f}" for lab, v in zip(labels, row_strae)), flush=True)
        st_rae_results.append(row_strae)

    st_rae_results = np.array(st_rae_results)
    print(f"\n  --- {isoform} full metric table, mean +/- std over {N_REPEATS} repeats, tuned LightGBM ---")
    header = f"  {'model':20s}" + "".join(f"{m:>14s}" for m in METRIC_NAMES)
    print(header)
    for lab in labels:
        row = f"  {lab:20s}"
        for m in METRIC_NAMES:
            vals = np.array(all_results[lab][m])
            row += f"{vals.mean():>8.4f}±{vals.std():.3f}".rjust(14)
        print(row)

    d = st_rae_results[:, 1] - st_rae_results[:, 0]
    t, p = ttest_rel(st_rae_results[:, 1], st_rae_results[:, 0])
    print(f"  {isoform} pharma vs baseline: delta={d.mean():+.4f} std={d.std():.4f}  "
          f"t={t:.3f} p={p:.4f}  same-dir={(d<0).sum()}/{N_REPEATS}")


def main() -> None:
    tr_all = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    for isoform, spec in ISOFORMS.items():
        run_isoform(isoform, spec, tr_all)


if __name__ == "__main__":
    main()
