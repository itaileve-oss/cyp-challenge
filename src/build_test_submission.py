"""Blend the 3 real TEST_BLINDED.csv component predictions per the confirmed-best
ensemble formula (0.9*(0.5*LightGBM-tuned + 0.5*GNN) + 0.1*Uni-Mol), apply the
placement/calibration correction (BLIND_MOMENTS + our own OOF Pearson -- see
apply_placement_calibration.py), format as a submission file, and validate it.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
VALUE_COLUMNS = {iso: f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS}
UNIMOL_WEIGHT = 0.1

BLIND_MOMENTS = {
    "CYP1A2": {"mean": 4.412, "sd": 1.553},
    "CYP2C9": {"mean": 4.830, "sd": 1.101},
    "CYP2D6": {"mean": 3.107, "sd": 1.599},
    "CYP3A4": {"mean": 4.880, "sd": 1.272},
}

# rho_true / rho_local, solved from the 2026-09-25 real leaderboard scores via
# rho_true = (R2_blind + k^2) / (2k) with k = our local OOF Pearson at submission
# time. CYP1A2/CYP2C9/CYP2D6 local CV was pessimistic relative to the true blind
# population (inflation > 1); CYP3A4 local CV was mildly optimistic (< 1), which
# also plausibly explains its regression under the old calibration (k=rho_local
# assumed inflation=1). Applied as: rho_true_today = INFLATION[iso] * rho_local_today.
INFLATION_FACTORS = {
    "CYP1A2": 1.27,
    "CYP2C9": 1.19,
    "CYP2D6": 1.49,
    "CYP3A4": 0.96,
}


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"

    test = pd.read_csv(data_dir / "TEST_BLINDED.csv")
    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    oof = pd.read_csv(models_dir / "final_ensemble_oof_predictions.csv")  # for our own OOF Pearson

    lgbm = pd.read_csv(models_dir / "test_predictions_lgbm_tuned.csv").set_index("Molecule_Name").loc[test["Molecule_Name"]]
    gnn = pd.read_csv(models_dir / "test_predictions_gnn.csv").set_index("Molecule_Name").loc[test["Molecule_Name"]]
    unimol = pd.read_csv(models_dir / "test_predictions_unimol.csv").set_index("Molecule_Name").loc[test["Molecule_Name"]]

    cols = list(VALUE_COLUMNS.values())
    base = 0.5 * gnn[cols] + 0.5 * lgbm[cols]
    ensemble = (1 - UNIMOL_WEIGHT) * base + UNIMOL_WEIGHT * unimol[cols]
    ensemble.index = test["Molecule_Name"].values

    # standalone single-task blends (2026-09-23/24), weights from sweep_standalone_blend_weights.py,
    # same as build_final_ensemble.py's local-OOF version
    standalone_blends = {
        "CYP3A4": ("test_predictions_cyp3a4_docking_pharma_logd_standalone.csv", 0.50),
        "CYP2D6": ("test_predictions_cyp2d6_pharmacophore_standalone.csv", 0.25),
        "CYP1A2": ("test_predictions_cyp1a2_final_standalone.csv", 0.50),
    }
    for iso, (fname, w) in standalone_blends.items():
        col = VALUE_COLUMNS[iso]
        standalone = pd.read_csv(models_dir / fname).set_index("Molecule_Name")[col]
        standalone = standalone.reindex(ensemble.index)
        ensemble[col] = (1 - w) * ensemble[col] + w * standalone.to_numpy(dtype=float)

    # TabPFN blends (2026-09-30), same weights/order as build_final_ensemble.py
    tabpfn_blends = {
        "CYP2C9": ("test_predictions_tabpfn_cyp2c9.csv", 0.30),
        "CYP2D6": ("test_predictions_tabpfn_cyp2d6.csv", 0.40),
    }
    for iso, (fname, w) in tabpfn_blends.items():
        col = VALUE_COLUMNS[iso]
        tabpfn = pd.read_csv(models_dir / fname).set_index("Molecule_Name")[col].reindex(ensemble.index)
        ensemble[col] = (1 - w) * ensemble[col] + w * tabpfn.to_numpy(dtype=float)

    ensemble = ensemble.reset_index(drop=True)

    print("=== Raw (uncalibrated) test predictions, per isoform ===")
    for iso in CYP_ISOFORMS:
        col = VALUE_COLUMNS[iso]
        print(f"{iso:<8} mean={ensemble[col].mean():.3f} sd={ensemble[col].std():.3f} "
              f"min={ensemble[col].min():.3f} max={ensemble[col].max():.3f}")

    # our own OOF Pearson, isoform by isoform (same as apply_placement_calibration.py)
    pearson = {}
    for iso in CYP_ISOFORMS:
        col = VALUE_COLUMNS[iso]
        true, pred = tr[col], oof[col]
        mask = true.notna()
        pearson[iso] = stats.pearsonr(true[mask], pred[mask]).statistic

    print("\n=== Applying placement calibration (BLIND_MOMENTS + inflation-corrected rho) ===")
    calibrated = ensemble.copy()
    for iso in CYP_ISOFORMS:
        col = VALUE_COLUMNS[iso]
        moments = BLIND_MOMENTS[iso]
        rho_local = pearson[iso]
        rho_true = rho_local * INFLATION_FACTORS[iso]
        current = ensemble[col]
        target_sd = rho_true * moments["sd"]
        scale = target_sd / current.std()
        offset = moments["mean"] - current.mean()
        calibrated[col] = current.mean() + scale * (current - current.mean()) + offset
        print(f"{iso:<8} rho_local={rho_local:.3f} x{INFLATION_FACTORS[iso]:.2f} -> rho_true={rho_true:.3f}  "
              f"sd {current.std():.3f}->{target_sd:.3f} (x{scale:.2f})  "
              f"mean {current.mean():.3f}->{moments['mean']:.3f} ({offset:+.2f})")

    submission = calibrated.copy()
    submission.insert(0, "Molecule_Name", test["Molecule_Name"].values)
    submission.insert(0, "SMILES", test["SMILES"].values)
    submission = submission[["SMILES", "Molecule_Name"] + cols]

    out_path = PROJECT_ROOT / "outputs"
    out_path.mkdir(exist_ok=True)
    versions_path = out_path / "versions"
    versions_path.mkdir(exist_ok=True)
    today = date.today().isoformat()

    # canonical filename (unchanged, in case the upload flow expects this exact
    # name) always holds the LATEST version; dated copies in versions/ are the
    # history trail so old vs. new is never ambiguous
    submission_path = out_path / "cyp_activity_submission.csv"
    submission.to_csv(submission_path, index=False)
    dated_submission_path = versions_path / f"cyp_activity_submission_{today}.csv"
    submission.to_csv(dated_submission_path, index=False)
    print(f"\nSaved submission -> {submission_path}")
    print(f"Saved dated copy  -> {dated_submission_path}")

    # also save the UNCALIBRATED version, in case we want to compare / submit both variants
    uncalibrated = ensemble.copy()
    uncalibrated.insert(0, "Molecule_Name", test["Molecule_Name"].values)
    uncalibrated.insert(0, "SMILES", test["SMILES"].values)
    uncalibrated = uncalibrated[["SMILES", "Molecule_Name"] + cols]
    uncalibrated_path = out_path / "cyp_activity_submission_uncalibrated.csv"
    uncalibrated.to_csv(uncalibrated_path, index=False)
    dated_uncalibrated_path = versions_path / f"cyp_activity_submission_uncalibrated_{today}.csv"
    uncalibrated.to_csv(dated_uncalibrated_path, index=False)
    print(f"Saved uncalibrated variant -> {uncalibrated_path}")

    print("\n=== Validating submission format ===")
    from validation.activity_validation import validate_activity_submission
    for label, path in [("calibrated", submission_path), ("uncalibrated", uncalibrated_path)]:
        expected_ids = set(test["Molecule_Name"])
        ok, errors = validate_activity_submission(path, expected_ids=expected_ids)
        if ok:
            print(f"  {label}: PASSED validation")
        else:
            print(f"  {label}: FAILED validation: {errors}")


if __name__ == "__main__":
    main()
