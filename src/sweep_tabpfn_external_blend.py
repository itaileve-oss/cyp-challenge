"""Score the 2026-09-30 TabPFN (CYP1A2/2C9/2D6) and external-stack (all 4)
OOF predictions against the current submitted ensemble
(models/final_ensemble_oof_predictions.csv, the 2026-09-24 composition), and
sweep a per-isoform blend weight for each component on top of it.

Point estimates (no bootstrap) for speed -- ST-RAE via the official
rae_soft_threshold_absolute_error, plus MAE/R2/Pearson/Spearman/Kendall.
Single split, single repeat: deltas near the ~0.005-0.01 noise floor need
repeated-CV confirmation before going into a submission.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))

from evaluation.custom_scoring_functions import rae_soft_threshold_absolute_error  # noqa: E402

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
MODELS = PROJECT_ROOT / "models"


def metrics(y, lo, hi, p):
    return {
        "ST-RAE": rae_soft_threshold_absolute_error(y, p, y_true_upper=hi, y_true_lower=lo),
        "MAE": mean_absolute_error(y, p),
        "R2": r2_score(y, p),
        "Pearson": pearsonr(y, p)[0],
        "Spearman": spearmanr(y, p)[0],
        "Kendall": kendalltau(y, p)[0],
    }


def load_oof(fname, endpoint):
    path = MODELS / fname
    if not path.exists():
        return None
    return pd.read_csv(path).set_index("Molecule_Name")[endpoint]


def main():
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    base_all = pd.read_csv(MODELS / "final_ensemble_oof_predictions.csv").set_index("Molecule_Name")
    rows = []
    for iso in ISOFORMS:
        ep = f"{iso}_pIC50_direct_inhibition"
        t = tr[tr[ep].notna()].set_index("Molecule_Name")
        y, lo, hi = t[ep].to_numpy(), t[f"{ep}_conf_low"].to_numpy(), t[f"{ep}_conf_high"].to_numpy()
        base = base_all.loc[t.index, ep].to_numpy()
        rows.append({"isoform": iso, "model": "ensemble (current)", "w": 0.0, **metrics(y, lo, hi, base)})

        comps = {
            "tabpfn": load_oof(f"cv_tabpfn_{iso.lower()}_oof_predictions.csv", ep),
            "ext_stack": load_oof(f"cv_{iso.lower()}_external_stack_oof_predictions.csv", ep),
        }
        for name, s in comps.items():
            if s is None:
                continue
            comp = s.reindex(t.index).to_numpy()
            if np.isnan(comp).any():
                print(f"WARNING {iso} {name}: {np.isnan(comp).sum()} missing OOF preds, skipping")
                continue
            rows.append({"isoform": iso, "model": f"{name} alone", "w": 1.0, **metrics(y, lo, hi, comp)})
            sweep = [(w, metrics(y, lo, hi, (1 - w) * base + w * comp)) for w in WEIGHTS[1:-1]]
            best_w, best_m = min(sweep, key=lambda r: r[1]["ST-RAE"])
            print(f"{iso} {name}: " + "  ".join(f"w={w:.1f}:{m['ST-RAE']:.4f}" for w, m in sweep))
            rows.append({"isoform": iso, "model": f"ens+{name} best", "w": best_w, **best_m})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(MODELS / "sweep_tabpfn_external_blend_results.csv", index=False)


if __name__ == "__main__":
    main()
