"""TabPFN extended to CYP1A2/CYP2C9/CYP2D6 (CYP3A4 already done in
cv_tabpfn_vs_lightgbm.py, 2026-09-23: TabPFN ST-RAE 0.5249 vs TUNED-LGBM
0.5336 / BEST-LGBM 0.5079). Same compact-feature-set rationale (no Morgan
fingerprint -- ~4x slower for TabPFN, poor fit for its pretrained continuous
prior), using each isoform's own established best local feature recipe minus
the fingerprint:
  CYP1A2: desc + docking(3) + pharma_v3(4)          (cv_cyp1a2_final_singletask.py)
  CYP2D6: desc + docking_best(3) + pharma_cyp2d6(2) (cv_cyp2d6_final_singletask.py)
  CYP2C9: desc only -- no isoform-specific extras exist; all 16 tried
          approaches were confirmed null (see outputs/versions/CHANGELOG.md),
          so CYP2C9's own best recipe IS the shared multitask base (fp+desc).

Run fold-by-fold (--isoform --fold) so each invocation fits under a single
tool-call wall-clock budget (~7-10 min/fold on this machine's GPU); a
--combine pass stitches the 5 per-isoform fold files into one OOF file.
GroupKFold has no random_state and is order-deterministic, so calling this
script separately per fold reproduces the exact same split every time.

Scope note: 1 repeat (not the 3 used for CYP3A4) for these three isoforms --
3 repeats x 3 isoforms x 5 folds x ~450s/fold ~= 5.6h of GPU time is not
practical to run fold-by-fold under this tool's constraints. This still
satisfies the actual decision-relevant goal (full-coverage OOF predictions
for blend-weight sweeping); it's a single-repeat, not full-rigor, metric read.
"""

import argparse
import pickle
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

from features import rdkit_descriptor_matrix  # noqa: E402
from cv_repeated_docking_other_isoforms import build_docking_isoform  # noqa: E402
from features_pharmacophore_cyp1a2_v3 import cyp1a2_pharmacophore_v3_matrix  # noqa: E402
from cv_repeated_other_isoform_pharmacophore import build_mols  # noqa: E402
from features_pharmacophore_other_isoforms import cyp2d6_feature_matrix  # noqa: E402
from cv_receptor_aggregation_other_isoforms import build_docking_best  # noqa: E402

N_FOLDS = 5
FOLD_DIR = PROJECT_ROOT / "models" / "tabpfn_fold_cache"
FOLD_DIR.mkdir(exist_ok=True)


def build_features(isoform: str, tr: pd.DataFrame):
    smiles = tr["SMILES"].tolist()
    names = tr["Molecule_Name"].tolist()
    desc = rdkit_descriptor_matrix(smiles)

    if isoform == "CYP1A2":
        dock = build_docking_isoform("CYP1A2", tr)
        with open(PROJECT_ROOT / "models" / "cyp1a2_winning_docked_poses.pkl", "rb") as f:
            poses = pickle.load(f)
        pharma = cyp1a2_pharmacophore_v3_matrix(names, smiles, poses)
        X = np.hstack([desc, dock, pharma])
    elif isoform == "CYP2D6":
        dock = build_docking_best("CYP2D6", tr)
        mols = build_mols(smiles, needs_conf=True)
        pharma = cyp2d6_feature_matrix(mols)
        X = np.hstack([desc, dock, pharma])
    elif isoform == "CYP2C9":
        X = desc
    else:
        raise ValueError(isoform)
    return X


def load_train(isoform: str) -> pd.DataFrame:
    endpoint = f"{isoform}_pIC50_direct_inhibition"
    tr_full = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    tr = tr_full[tr_full[endpoint].notna()].reset_index(drop=True)
    return tr


def get_fold_split(isoform: str, tr: pd.DataFrame):
    clusters = np.array(uru.get_bemis_murcko_clusters(tr["SMILES"].tolist()))
    gkf = GroupKFold(n_splits=N_FOLDS)
    X_dummy = np.zeros((len(tr), 1))
    y = tr[f"{isoform}_pIC50_direct_inhibition"].to_numpy(dtype=float)
    return list(gkf.split(X_dummy, y, groups=clusters))


def run_one_fold(isoform: str, fold: int) -> None:
    tr = load_train(isoform)
    endpoint = f"{isoform}_pIC50_direct_inhibition"
    y = tr[endpoint].to_numpy(dtype=float)

    print(f"[{isoform} fold {fold}] featurizing...", flush=True)
    X = build_features(isoform, tr)
    print(f"[{isoform} fold {fold}] X shape={X.shape}", flush=True)

    splits = get_fold_split(isoform, tr)
    tr_idx, te_idx = splits[fold]

    t0 = time.time()
    model = TabPFNRegressor(device="cuda", random_state=42)
    model.fit(X[tr_idx], y[tr_idx])
    preds = model.predict(X[te_idx])
    dt = time.time() - t0
    print(f"[{isoform} fold {fold}] done in {dt:.1f}s, n_test={len(te_idx)}", flush=True)

    out = {
        "test_idx": te_idx,
        "molecule_names": tr["Molecule_Name"].to_numpy()[te_idx],
        "preds": preds,
        "wall_time_s": dt,
    }
    out_path = FOLD_DIR / f"{isoform}_fold{fold}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"[{isoform} fold {fold}] saved -> {out_path}", flush=True)


def combine(isoform: str) -> pd.DataFrame:
    tr = load_train(isoform)
    endpoint = f"{isoform}_pIC50_direct_inhibition"
    oof = np.full(len(tr), np.nan)
    total_time = 0.0
    for fold in range(N_FOLDS):
        path = FOLD_DIR / f"{isoform}_fold{fold}.pkl"
        with open(path, "rb") as f:
            d = pickle.load(f)
        oof[d["test_idx"]] = d["preds"]
        total_time += d["wall_time_s"]
    predictions = pd.DataFrame({"Molecule_Name": tr["Molecule_Name"], endpoint: oof})
    print(f"[{isoform}] combined {N_FOLDS} folds, total TabPFN wall time={total_time:.1f}s", flush=True)
    return predictions


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isoform", required=True, choices=["CYP1A2", "CYP2C9", "CYP2D6"])
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    if args.combine:
        df = combine(args.isoform)
        out_path = PROJECT_ROOT / "models" / f"cv_tabpfn_{args.isoform.lower()}_oof_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
    else:
        run_one_fold(args.isoform, args.fold)
