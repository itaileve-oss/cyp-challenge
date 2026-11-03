"""Error analysis: correlate CYP3A4 docking score with true pIC50, and find the
worst false-negative cases (true pIC50 high, model badly underpredicted) --
then look at their docking pose/contact features to see whether the model is
missing something the physical pose would have revealed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"

    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    preds = pd.read_csv(models_dir / "final_ensemble_oof_predictions.csv")
    dock = pd.read_csv(models_dir / "cyp3a4_docking_scores.csv")
    contacts = pd.read_csv(models_dir / "cyp3a4_docking_contact_features.csv")

    df = tr[["Molecule_Name", "SMILES", "CYP3A4_pIC50_direct_inhibition",
             "CYP3A4_pIC50_direct_inhibition_conf_low", "CYP3A4_pIC50_direct_inhibition_conf_high"]].copy()
    df = df.rename(columns={"CYP3A4_pIC50_direct_inhibition": "true_pIC50"})
    df = df.merge(preds[["Molecule_Name", "CYP3A4_pIC50_direct_inhibition"]].rename(
        columns={"CYP3A4_pIC50_direct_inhibition": "pred_pIC50"}), on="Molecule_Name", how="inner")
    df = df.merge(dock, on="Molecule_Name", how="inner")
    df = df.merge(contacts, on="Molecule_Name", how="left")
    df = df.dropna(subset=["true_pIC50", "pred_pIC50", "vina_best_affinity"])

    print(f"Merged rows (has true pIC50 + prediction + docking score): {len(df)}")

    df["residual"] = df["true_pIC50"] - df["pred_pIC50"]

    r_dock_true, p1 = stats.pearsonr(df["vina_best_affinity"], df["true_pIC50"])
    rho_dock_true, p1s = stats.spearmanr(df["vina_best_affinity"], df["true_pIC50"])
    print(f"\nDocking affinity vs TRUE pIC50: Pearson r={r_dock_true:.3f} (p={p1:.2e}), "
          f"Spearman rho={rho_dock_true:.3f} (p={p1s:.2e})  [more negative kcal/mol = tighter predicted binding]")

    r_dock_pred, p2 = stats.pearsonr(df["vina_best_affinity"], df["pred_pIC50"])
    print(f"Docking affinity vs MODEL-PREDICTED pIC50: Pearson r={r_dock_pred:.3f} (p={p2:.2e})")

    r_dock_resid, p3 = stats.pearsonr(df["vina_best_affinity"], df["residual"])
    print(f"Docking affinity vs residual (true-pred): Pearson r={r_dock_resid:.3f} (p={p3:.2e})")

    df.to_csv(models_dir / "cyp3a4_docking_pic50_merged.csv", index=False)
    print(f"\nSaved merged table ({len(df)} rows) to models/cyp3a4_docking_pic50_merged.csv")

    print("\n=== TOP 15 FALSE NEGATIVES (true pIC50 >> predicted) ===")
    fn = df.sort_values("residual", ascending=False).head(15)
    cols_show = ["Molecule_Name", "true_pIC50", "pred_pIC50", "residual",
                 "vina_best_affinity", "vina_n_poses", "vina_affinity_spread",
                 "dock_fe_min_dist", "dock_ser119_min_dist", "dock_arg212_min_dist",
                 "dock_phe_cage_contacts", "SMILES"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 40)
    print(fn[cols_show].to_string(index=False))

    print("\n=== For comparison: population medians/means of docking features ===")
    print(df[["vina_best_affinity", "dock_fe_min_dist", "dock_ser119_min_dist",
               "dock_arg212_min_dist", "dock_phe_cage_contacts"]].describe())

    fn.to_csv(models_dir / "cyp3a4_top_false_negatives.csv", index=False)
    print("\nSaved top false negatives to models/cyp3a4_top_false_negatives.csv")


if __name__ == "__main__":
    main()
