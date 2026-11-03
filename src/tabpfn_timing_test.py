import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.model_selection import GroupKFold
from tabpfn import TabPFNRegressor

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import useful_rdkit_utils as uru  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from features import morgan_fp_matrix, rdkit_descriptor_matrix  # noqa: E402
from features_logd import logd_feature_matrix  # noqa: E402
from cv_repeated_combined_docking_pharma import build_docking, build_pharma_nan  # noqa: E402

ENDPOINT = "CYP3A4_pIC50_direct_inhibition"


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr[tr[ENDPOINT].notna()].reset_index(drop=True)
    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)

    print("Featurizing...", flush=True)
    fp = morgan_fp_matrix(smiles)
    desc = rdkit_descriptor_matrix(smiles)
    dock = build_docking(tr)
    pharma = build_pharma_nan(tr)
    logd = logd_feature_matrix(smiles)
    X_all = np.hstack([fp, desc, dock, pharma, logd])
    X_compact = np.hstack([desc, dock, pharma, logd])  # no fingerprint, ~211 cols
    print(f"X_all shape: {X_all.shape}, X_compact shape: {X_compact.shape}", flush=True)

    base_clusters = np.array(uru.get_bemis_murcko_clusters(smiles))
    unique_clusters = np.unique(base_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    cluster_int = np.array([cluster_to_idx[c] for c in base_clusters])

    rng = np.random.default_rng(123)
    perm = rng.permutation(len(unique_clusters))
    relabeled = perm[cluster_int]
    gkf = GroupKFold(n_splits=5)
    tr_idx, te_idx = next(iter(gkf.split(X_all, y, groups=relabeled)))
    print(f"fold sizes: train={len(tr_idx)} test={len(te_idx)}", flush=True)

    for name, X in [("compact (211 cols, no fp)", X_compact), ("all (2276 cols, with fp)", X_all)]:
        print(f"\n--- TabPFN on {name} ---", flush=True)
        t0 = time.time()
        model = TabPFNRegressor(device="cuda", ignore_pretraining_limits=True, random_state=42)
        model.fit(X[tr_idx], y[tr_idx])
        t_fit = time.time() - t0
        t1 = time.time()
        preds = model.predict(X[te_idx])
        t_pred = time.time() - t1
        rho, _ = spearmanr(preds, y[te_idx])
        print(f"  fit={t_fit:.1f}s predict={t_pred:.1f}s single-fold Spearman(sanity)={rho:.4f}", flush=True)


if __name__ == "__main__":
    main()
