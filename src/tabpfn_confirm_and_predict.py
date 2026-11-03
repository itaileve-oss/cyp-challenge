"""TabPFN for CYP2C9/CYP2D6 (2026-09-30): (1) real TEST_BLINDED.csv predictions
from a full-train fit, (2) repeated cluster-CV confirmation of the TabPFN blend
found by sweep_tabpfn_external_blend.py (2C9 w=0.3: 0.6344->0.6230, 2D6 w=0.4:
0.8802->0.8682, single split, weight picked on the same OOF).

Repeats reshuffle Bemis-Murcko cluster->fold assignment (same rng(123)
permutation scheme as cv_repeated_cyp2c9_robust_training.py). The ensemble OOF
(final_ensemble_oof_predictions.csv) is held fixed -- re-running GNN/Uni-Mol per
repeat is too expensive -- so this measures how stable the TabPFN gain is
across TabPFN's own fold assignments. Also reports a NESTED weight (chosen on
the other 4 folds, applied to the held-out one) to remove the weight-selection
optimism of the original sweep.

Features are built ONCE per isoform over train+test together (consistent
descriptor-column dropping) and cached. Every fold result is cached, so the
driver can be re-run after a crash and skips finished work.

Usage:
  --isoform X --featurize        build+cache features
  --isoform X --test             full-train fit -> models/test_predictions_tabpfn_{iso}.csv
  --isoform X --repeat R --fold F
  --analyze                      summary table over all cached repeats
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import useful_rdkit_utils as uru  # noqa: E402
from evaluation.custom_scoring_functions import rae_soft_threshold_absolute_error  # noqa: E402

N_FOLDS = 5
N_REPEATS = 3  # repeat 0 = original unshuffled GroupKFold split (cached in tabpfn_fold_cache/)
BLEND_W = {"CYP2C9": 0.3, "CYP2D6": 0.4, "CYP1A2": 0.4, "CYP3A4": 0.3}  # 1A2 from the 9/30 sweep; 3A4 provisional, the nested weight is what counts
W_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MODELS = PROJECT_ROOT / "models"
CACHE = MODELS / "tabpfn_confirm_cache"
CACHE.mkdir(exist_ok=True)
DATA = PROJECT_ROOT / "data" / "raw"


def endpoint(iso):
    return f"{iso}_pIC50_direct_inhibition"


def load_train(iso):
    tr = pd.read_csv(DATA / "TRAIN_inhibition.csv")
    return tr[tr[endpoint(iso)].notna()].reset_index(drop=True)


def featurize(iso):
    from features import rdkit_descriptor_matrix

    tr = load_train(iso)
    te = pd.read_csv(DATA / "TEST_BLINDED.csv")
    desc = rdkit_descriptor_matrix(tr["SMILES"].tolist() + te["SMILES"].tolist())
    X_tr, X_te = desc[: len(tr)], desc[len(tr):]
    if iso == "CYP2D6":
        from cv_receptor_aggregation_other_isoforms import build_docking_best
        from cv_repeated_other_isoform_pharmacophore import build_mols
        from features_pharmacophore_other_isoforms import cyp2d6_feature_matrix
        from predict_test_cyp2d6_final import build_test_docking

        dock_tr = build_docking_best("CYP2D6", tr)
        dock_te = build_test_docking(te)
        ph_tr = cyp2d6_feature_matrix(build_mols(tr["SMILES"].tolist(), needs_conf=True))
        ph_te = cyp2d6_feature_matrix(build_mols(te["SMILES"].tolist(), needs_conf=True))
        X_tr = np.hstack([X_tr, dock_tr, ph_tr])
        X_te = np.hstack([X_te, dock_te, ph_te])
    elif iso == "CYP1A2":
        from cv_repeated_docking_other_isoforms import build_docking_isoform
        from features_pharmacophore_cyp1a2_v3 import cyp1a2_pharmacophore_v3_matrix
        from predict_test_cyp1a2_final import build_test_docking

        poses_tr = pickle.load(open(MODELS / "cyp1a2_winning_docked_poses.pkl", "rb"))
        poses_te = pickle.load(open(MODELS / "test_cyp1a2_winning_docked_poses.pkl", "rb"))
        X_tr = np.hstack([X_tr, build_docking_isoform("CYP1A2", tr),
                          cyp1a2_pharmacophore_v3_matrix(tr["Molecule_Name"].tolist(), tr["SMILES"].tolist(), poses_tr)])
        X_te = np.hstack([X_te, build_test_docking(te),
                          cyp1a2_pharmacophore_v3_matrix(te["Molecule_Name"].tolist(), te["SMILES"].tolist(), poses_te)])
    elif iso == "CYP3A4":
        # compact set from cv_tabpfn_vs_lightgbm.py: desc + docking(3) + native-NaN pharma(5) + logD(3)
        from cv_repeated_combined_docking_pharma import build_docking, build_pharma_nan
        from features_logd import logd_feature_matrix
        from predict_test_cyp3a4_docking_pharma_logd import build_test_docking, build_test_pharma

        X_tr = np.hstack([X_tr, build_docking(tr), build_pharma_nan(tr), logd_feature_matrix(tr["SMILES"].tolist())])
        X_te = np.hstack([X_te, build_test_docking(te), build_test_pharma(te), logd_feature_matrix(te["SMILES"].tolist())])
    print(f"[{iso}] X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)
    with open(CACHE / f"{iso}_features.pkl", "wb") as f:
        pickle.dump({"X_tr": X_tr, "X_te": X_te, "names_te": te["Molecule_Name"].to_numpy()}, f)


def load_features(iso):
    with open(CACHE / f"{iso}_features.pkl", "rb") as f:
        return pickle.load(f)


def fit_predict(X_fit, y_fit, X_pred):
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(device="cuda", random_state=42)
    model.fit(X_fit, y_fit)
    return model.predict(X_pred)


def run_test(iso):
    out_path = MODELS / f"test_predictions_tabpfn_{iso.lower()}.csv"
    if out_path.exists():
        print(f"[{iso}] test preds exist, skipping", flush=True)
        return
    feats = load_features(iso)
    y = load_train(iso)[endpoint(iso)].to_numpy(dtype=float)
    t0 = time.time()
    preds = fit_predict(feats["X_tr"], y, feats["X_te"])
    pd.DataFrame({"Molecule_Name": feats["names_te"], endpoint(iso): preds}).to_csv(out_path, index=False)
    print(f"[{iso}] test preds done in {time.time()-t0:.0f}s -> {out_path}", flush=True)


def fold_splits(iso, tr, repeat):
    clusters = np.array(uru.get_bemis_murcko_clusters(tr["SMILES"].tolist()))
    groups = clusters
    if repeat > 0:
        uniq = np.unique(clusters)
        cint = np.searchsorted(uniq, clusters)
        rng = np.random.default_rng(123)
        for _ in range(repeat):
            perm = rng.permutation(len(uniq))
        groups = perm[cint]
    return list(GroupKFold(n_splits=N_FOLDS).split(np.zeros((len(tr), 1)), groups=groups))


def run_fold(iso, repeat, fold):
    out_path = CACHE / f"{iso}_r{repeat}_f{fold}.pkl"
    if out_path.exists():
        print(f"[{iso} r{repeat} f{fold}] cached, skipping", flush=True)
        return
    tr = load_train(iso)
    y = tr[endpoint(iso)].to_numpy(dtype=float)
    X = load_features(iso)["X_tr"]
    tr_idx, te_idx = fold_splits(iso, tr, repeat)[fold]
    t0 = time.time()
    preds = fit_predict(X[tr_idx], y[tr_idx], X[te_idx])
    with open(out_path, "wb") as f:
        pickle.dump({"test_idx": te_idx, "preds": preds}, f)
    print(f"[{iso} r{repeat} f{fold}] done in {time.time()-t0:.0f}s", flush=True)


def metrics(y, lo, hi, p):
    return {
        "ST-RAE": rae_soft_threshold_absolute_error(y, p, y_true_upper=hi, y_true_lower=lo),
        "MAE": mean_absolute_error(y, p), "R2": r2_score(y, p), "Pearson": pearsonr(y, p)[0],
        "Spearman": spearmanr(y, p)[0], "Kendall": kendalltau(y, p)[0],
    }


def load_repeat_oof(iso, repeat, n):
    oof, fold_of = np.full(n, np.nan), np.full(n, -1)
    for fold in range(N_FOLDS):
        path = CACHE / f"{iso}_r{repeat}_f{fold}.pkl"
        if repeat == 0 and not path.exists():
            path = MODELS / "tabpfn_fold_cache" / f"{iso}_fold{fold}.pkl"
        if not path.exists():
            return None, None
        with open(path, "rb") as f:
            d = pickle.load(f)
        oof[d["test_idx"]], fold_of[d["test_idx"]] = d["preds"], fold
    return oof, fold_of


def analyze():
    ens_all = pd.read_csv(MODELS / "final_ensemble_oof_predictions.csv").set_index("Molecule_Name")
    rows = []
    for iso in BLEND_W:
        ep = endpoint(iso)
        tr = load_train(iso)
        y, lo, hi = (tr[c].to_numpy(dtype=float) for c in [ep, f"{ep}_conf_low", f"{ep}_conf_high"])
        ens = ens_all.loc[tr["Molecule_Name"], ep].to_numpy(dtype=float)
        base = metrics(y, lo, hi, ens)
        for rep in range(N_REPEATS):
            tab, fold_of = load_repeat_oof(iso, rep, len(tr))
            if tab is None:
                print(f"[{iso} r{rep}] incomplete, skipped")
                continue
            fixed = metrics(y, lo, hi, (1 - BLEND_W[iso]) * ens + BLEND_W[iso] * tab)
            # nested: pick w on the other 4 folds, apply to the held-out fold
            nested = np.empty(len(tr))
            chosen = []
            for f in range(N_FOLDS):
                o, h = fold_of != f, fold_of == f
                w = min(W_GRID, key=lambda w: rae_soft_threshold_absolute_error(
                    y[o], (1 - w) * ens[o] + w * tab[o], y_true_upper=hi[o], y_true_lower=lo[o]))
                chosen.append(w)
                nested[h] = (1 - w) * ens[h] + w * tab[h]
            nest = metrics(y, lo, hi, nested)
            for label, m in [("tabpfn alone", metrics(y, lo, hi, tab)), (f"fixed w={BLEND_W[iso]}", fixed),
                             (f"nested w={chosen}", nest)]:
                rows.append({"isoform": iso, "repeat": rep, "model": label,
                             **{k: v for k, v in m.items()},
                             "dST-RAE": m["ST-RAE"] - base["ST-RAE"]})
        rows.append({"isoform": iso, "repeat": "-", "model": "ensemble (current)", **base, "dST-RAE": 0.0})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(MODELS / "tabpfn_confirm_results.csv", index=False)
    print("\n=== mean +/- std over repeats (delta vs current ensemble) ===")
    df2 = df[df.repeat != "-"].copy()
    df2["kind"] = df2.model.str.split(" w=").str[0]
    for (iso, kind), g in df2.groupby(["isoform", "kind"]):
        print(f"{iso} {kind:13s} n={len(g)}  dST-RAE={g['dST-RAE'].mean():+.4f} +/- {g['dST-RAE'].std():.4f}  "
              + "  ".join(f"{k}={g[k].mean():.4f}" for k in ["ST-RAE", "MAE", "R2", "Pearson", "Spearman", "Kendall"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--isoform", choices=list(BLEND_W))
    ap.add_argument("--featurize", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--repeat", type=int)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.featurize:
        if not (CACHE / f"{a.isoform}_features.pkl").exists():
            featurize(a.isoform)
    elif a.test:
        run_test(a.isoform)
    else:
        run_fold(a.isoform, a.repeat, a.fold)
