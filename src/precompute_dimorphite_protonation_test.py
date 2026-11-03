"""Extends models/dimorphite_ph74.csv (built by precompute_dimorphite_protonation.py
for all 4905 TRAIN_inhibition.csv compounds) to also cover TEST_BLINDED.csv's 750
compounds, so features_logd.py's cache lookup (keyed by SMILES) works for the blind
test set too -- needed for the CYP3A4 logD feature to be usable in the actual
submission, not just local CV. Appends rather than overwrites.
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
    existing = pd.read_csv(OUT_CSV)
    test = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "TEST_BLINDED.csv")
    smiles_list = test["SMILES"].tolist()

    already = set(existing["SMILES"])
    new_smiles = [s for s in smiles_list if s not in already]
    print(f"{len(smiles_list)} TEST_BLINDED compounds, {len(new_smiles)} not already cached", flush=True)

    rows = []
    n_failed = 0
    for i, smi in enumerate(new_smiles):
        prot = protonate_at_74(smi)
        if prot is None:
            n_failed += 1
            prot = smi
        rows.append((smi, prot))
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(new_smiles)}] failed={n_failed}", flush=True)

    print(f"DONE: {len(new_smiles)-n_failed}/{len(new_smiles)} succeeded", flush=True)
    new_df = pd.DataFrame(rows, columns=["SMILES", "protonated_smiles_ph74"])
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(combined)} total rows)")


if __name__ == "__main__":
    main()
