"""Same baseline-vs-docked-electrostatics GNN A/B as cv_cyp3a4_docked_gnn.py, but
restricted to ONLY the compounds where the docked-pose atom-matching + xtb single
point actually succeeded (nonzero feature vector) -- both arms trained/evaluated
on the exact same subset, so a fair comparison isn't confounded by ~48% of
compounds carrying a fabricated all-zero placeholder instead of a real feature.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import useful_rdkit_utils as uru

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cv_cyp3a4_docked_gnn import ENDPOINT, DOCKED_ELEC_CACHE, run, report  # noqa: E402


def main() -> None:
    tr_full = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    with open(DOCKED_ELEC_CACHE, "rb") as f:
        elec_full = pickle.load(f)

    succeeded = np.array([feat.any() for feat in elec_full])
    tr_cyp3a4 = tr_full[tr_full[ENDPOINT].notna()]
    print(f"CYP3A4-labeled: {len(tr_cyp3a4)}, of which xtb succeeded: "
          f"{succeeded[tr_full[ENDPOINT].notna().to_numpy()].sum()}")

    keep_mask = tr_full[ENDPOINT].notna().to_numpy() & succeeded
    tr = tr_full[keep_mask].reset_index(drop=True)
    elec_subset = [feat for feat, k in zip(elec_full, keep_mask) if k]
    print(f"Clean subset (both arms use this exact set): {len(tr)} compounds", flush=True)

    smiles = tr["SMILES"].tolist()
    y = tr[ENDPOINT].to_numpy(dtype=float)
    clusters = np.array(uru.get_bemis_murcko_clusters(smiles))

    print("\n=== Baseline (topology + rotatable-bond flag, no electrostatics), CLEAN subset ===", flush=True)
    oof_base = run(smiles, y, clusters, extra_atom_features_list=None)
    report(oof_base, tr, "baseline, clean subset")

    print("\n=== +docked electrostatics, CLEAN subset (same compounds) ===", flush=True)
    oof_elec = run(smiles, y, clusters, extra_atom_features_list=elec_subset)
    report(oof_elec, tr, "+docked electrostatics, clean subset")

    out = pd.DataFrame({
        "Molecule_Name": tr["Molecule_Name"],
        f"{ENDPOINT}_baseline": oof_base,
        f"{ENDPOINT}_docked_elec": oof_elec,
    })
    out.to_csv(PROJECT_ROOT / "models" / "cv_cyp3a4_docked_gnn_clean_oof_predictions.csv", index=False)
    print("\nSaved to models/cv_cyp3a4_docked_gnn_clean_oof_predictions.csv")


if __name__ == "__main__":
    main()
