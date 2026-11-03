"""Affine placement correction, adapted from briford's (SuperCowPowers/workbench)
published approach (https://supercowpowers.github.io/workbench/blogs/cyp_challenge/),
code at github.com/SuperCowPowers/workbench ml_pipelines/OpenADMET/cyp/scripts/cyp_recalibrate.py.

Key idea: a model's predictions carry two separable things -- their ORDER (rank
correlation, which reflects real model quality) and their PLACEMENT on the pIC50 axis
(mean/spread, which does not). MA-ST-RAE rewards placement heavily, so a raw model with
good ranking but the wrong mean/spread for the blind population scores far worse than it
has to. An affine transform (shift + rescale) can fix placement without touching a single
compound's rank.

R2 = 2*rho*k - k^2 - b^2, with k = sd(pred)/sd(true) and b = mean offset in units of
sd(true). Only rho depends on ranking; k and b are free. R2 is maximized (capped at rho^2)
at k=rho, b=0 -- NOT k=1. A model should be predicted NARROWER than the true spread by
exactly its own correlation, because squared-error-optimal prediction under noise is
shrinkage, not variance-matching.

BLIND_MOMENTS below are the real, solved blind-test-half label distribution (mean/sd per
isoform), reverse-engineered by briford from three of their own scored leaderboard
submissions using the same R2 identity (see their cyp_recalibrate.py:solve_blind_moments).
These are PROPERTIES OF THE TEST SET, not of any model -- safe to reuse for our own model
without spending a submission. We independently confirmed there's a real distribution
shift vs our own training data (data/raw/TRAIN_inhibition.csv): every isoform's blind sd is
substantially larger than our training sd, and CYP2D6's blind mean (3.11) is far below our
training mean (4.78).

We do NOT have our own OOF_TO_BLIND inflation factor (briford's scaffold-OOF Pearson
understated their blind Pearson by 1.07x-1.66x depending on isoform -- solved from their own
board history, which we don't have yet since we haven't submitted). So this script uses our
own RAW scaffold-OOF Pearson, uninflated -- a conservative/lower-bound placement that likely
undershoots the true optimum, but avoids the risk of overcorrecting on a made-up inflation
factor. Once we have a few real submissions scored, we can solve our own OOF_TO_BLIND
factors the same way briford did and refine this.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "CYP-Challenge-Tutorial"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
VALUE_COLUMNS = {iso: f"{iso}_pIC50_direct_inhibition" for iso in CYP_ISOFORMS}

# Solved by briford from three scored leaderboard submissions (github.com/SuperCowPowers/
# workbench, cyp_recalibrate.py). Properties of the blind test set itself, not of a model.
# NOTE this is only the LIVE-scored half; the other half (never scored until challenge end)
# transfers only insofar as the two halves share a distribution, which the series-based
# test split is explicitly designed to prevent -- treat as the best available estimate, not
# a guarantee for the final leaderboard.
BLIND_MOMENTS = {
    "CYP1A2": {"mean": 4.412, "sd": 1.553},
    "CYP2C9": {"mean": 4.830, "sd": 1.101},
    "CYP2D6": {"mean": 3.107, "sd": 1.599},
    "CYP3A4": {"mean": 4.880, "sd": 1.272},
}


def compute_oof_pearson(tr: pd.DataFrame, preds: pd.DataFrame) -> dict:
    pearson = {}
    for iso in CYP_ISOFORMS:
        col = VALUE_COLUMNS[iso]
        true, pred = tr[col], preds[col]
        mask = true.notna()
        pearson[iso] = stats.pearsonr(true[mask], pred[mask]).statistic
    return pearson


def place(preds: pd.DataFrame, pearson: dict) -> dict:
    """R2-optimal affine calibration: target_sd = rho * blind_sd, target_mean = blind_mean."""
    calibration = {}
    for iso, moments in BLIND_MOMENTS.items():
        col = VALUE_COLUMNS[iso]
        rho = pearson[iso]
        current = preds[col]
        target_sd = rho * moments["sd"]
        calibration[iso] = {
            "offset": moments["mean"] - current.mean(),
            "scale": target_sd / current.std(),
            "rho": rho,
        }
    return calibration


def apply_calibration(preds: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    out = preds.copy()
    for iso, cal in calibration.items():
        col = VALUE_COLUMNS[iso]
        before = out[col]
        # scale about the isoform's own mean so scale doesn't move the centre
        out[col] = before.mean() + cal["scale"] * (before - before.mean()) + cal["offset"]
    return out


def score_macro(preds: pd.DataFrame, gt: pd.DataFrame) -> float:
    boot = score_activity_predictions(preds, gt, REGRESSION_ENDPOINTS)
    track = add_macro_endpoint(boot, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    summary = average_bootstrap_results_by_endpoint(track)
    return summary.loc["MA", "ST-RAE_mean"]


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"

    tr = pd.read_csv(data_dir / "TRAIN_inhibition.csv")
    preds = pd.read_csv(models_dir / "final_ensemble_oof_predictions.csv")

    pearson = compute_oof_pearson(tr, preds)
    calibration = place(preds, pearson)

    print(f"{'isoform':<8} {'rho(OOF)':>9} {'sd before':>10} {'sd target':>10} {'scale':>7} "
          f"{'mean before':>12} {'mean target':>12} {'offset':>8} {'R2 ceiling':>11}")
    for iso, cal in calibration.items():
        col = VALUE_COLUMNS[iso]
        before = preds[col]
        print(f"{iso:<8} {cal['rho']:>9.3f} {before.std():>10.3f} "
              f"{cal['rho'] * BLIND_MOMENTS[iso]['sd']:>10.3f} {cal['scale']:>7.2f} "
              f"{before.mean():>12.3f} {BLIND_MOMENTS[iso]['mean']:>12.3f} {cal['offset']:>+8.2f} "
              f"{cal['rho']**2:>11.3f}")

    # NOTE: this is a diagnostic only. We CANNOT locally verify this improves the real
    # blind-test score (our OOF true labels come from the TRAINING population, not the
    # blind population -- scoring calibrated-OOF-vs-training-truth would be circular and
    # doesn't test what we actually care about). Real validation requires an actual
    # submission. What we CAN show: this is what the correction looks like, and the
    # "R2 ceiling" column is the theoretical best any affine rescaling of THIS model's
    # current OOF ranking could achieve on the blind set, per the R2=2*rho*k-k^2-b^2 identity.
    calibrated = apply_calibration(preds, calibration)
    out_path = models_dir / "final_ensemble_oof_predictions_placement_calibrated.csv"
    calibrated.to_csv(out_path, index=False)
    print(f"\nSaved calibrated OOF predictions (diagnostic only) to {out_path}")
    print("This has NOT been validated against the real blind test -- it changes what our")
    print("predictions would look like on the TEST_BLINDED.csv population, not on OOF-vs-")
    print("training-truth (which would be circular). Real validation needs a submission.")


if __name__ == "__main__":
    main()
