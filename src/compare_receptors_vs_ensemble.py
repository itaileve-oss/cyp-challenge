"""Compare each individual receptor conformation against ensemble aggregation
(best-of-all / mean-of-all across the 5 receptors), for docking energy and the
heme-term correlation against true CYP3A4 pIC50, plus Fe-coordination rate.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
raw = pd.read_csv(PROJECT_ROOT / "models" / "gd2heme_ensemble_raw_merged.csv")

TARGET = "CYP3A4_pIC50_direct_inhibition"


def eval_group(df_agg: pd.DataFrame, label: str) -> dict:
    r, p = pearsonr(df_agg.docking_energy, df_agg[TARGET])
    rho, ps = spearmanr(df_agg.docking_energy, df_agg[TARGET])
    r_hm, _ = pearsonr(df_agg.hm_e, df_agg[TARGET])
    coord_frac = df_agg.fe_coordinating.mean()
    return {"label": label, "n": len(df_agg), "pearson_energy": r, "pearson_p": p,
            "spearman": rho, "pearson_hm_e": r_hm, "fe_coord_frac": coord_frac}


rows = []

# Per-receptor, best-of-stereo-variants aggregation
for receptor, g in raw.groupby("receptor"):
    best = g.loc[g.groupby("Molecule_Name").docking_energy.idxmin()]
    rows.append(eval_group(best, f"{receptor} (best-of-stereo)"))

# Per-receptor, mean-of-stereo-variants aggregation
for receptor, g in raw.groupby("receptor"):
    mean_g = g.groupby(["Molecule_Name", TARGET], as_index=False).agg(
        docking_energy=("docking_energy", "mean"), hm_e=("hm_e", "mean"),
        fe_coordinating=("fe_coordinating", "mean"),
    )
    rows.append(eval_group(mean_g, f"{receptor} (mean-of-stereo)"))

# Ensemble: best-of-all (min across all 5 receptors x all stereo variants)
best_overall = raw.loc[raw.groupby("Molecule_Name").docking_energy.idxmin()]
rows.append(eval_group(best_overall, "ENSEMBLE (best-of-all)"))

# Ensemble: mean-of-all
mean_overall = raw.groupby(["Molecule_Name", TARGET], as_index=False).agg(
    docking_energy=("docking_energy", "mean"), hm_e=("hm_e", "mean"),
    fe_coordinating=("fe_coordinating", "mean"),
)
rows.append(eval_group(mean_overall, "ENSEMBLE (mean-of-all)"))

summary = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(summary.to_string(index=False))
summary.to_csv(PROJECT_ROOT / "models" / "gd2heme_receptor_vs_ensemble_comparison.csv", index=False)
print("\nSaved to models/gd2heme_receptor_vs_ensemble_comparison.csv")
