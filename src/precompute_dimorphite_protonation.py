"""One-time precompute: pH-7.4 protonation state for every compound in
TRAIN_inhibition.csv via dimorphite-dl's curated per-substructure pKa rules
(site_substructures.smarts) -- a real, curated pKa lookup rather than the
2-class Henderson-Hasselbalch approximation in the first cv_repeated_logd.py
pass. Cached to disk because dimorphite-dl only lives in the GNINA conda env,
while the modeling/CV scripts run in oadmet_cyp_tutorial (has useful_rdkit_utils
/ lightgbm / evaluation, not dimorphite-dl) -- same env split as the docking
pipeline. Run this under GNINA; features_logd.py just reads the cached CSV.
"""

from pathlib import Path

import pandas as pd
from dimorphite_dl import protonate_smiles

PROJECT_ROOT = Path("/home/ileven/Projects/cyp-challenge")
OUT_CSV = PROJECT_ROOT / "models" / "dimorphite_ph74.csv"


def protonate_at_74(smiles: str) -> str | None:
    try:
        out = list(protonate_smiles(smiles, ph_min=7.4, ph_max=7.4, precision=0.0, max_variants=4))
    except Exception:
        return None
    return out[0] if out else None


def main() -> None:
    tr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TRAIN_inhibition.csv")
    smiles_list = tr["SMILES"].tolist()
    rows = []
    n_failed = 0
    for i, smi in enumerate(smiles_list):
        prot = protonate_at_74(smi)
        if prot is None:
            n_failed += 1
            prot = smi  # fall back to the neutral input
        rows.append((smi, prot))
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(smiles_list)}] failed={n_failed}", flush=True)

    print(f"DONE: {len(smiles_list)-n_failed}/{len(smiles_list)} succeeded", flush=True)
    out = pd.DataFrame(rows, columns=["SMILES", "protonated_smiles_ph74"])
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
