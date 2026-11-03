"""Parse the Vina-GPU docking output (tmp/docked_all/*.pdbqt, from the full
2335-ligand CYP3A4 GPU docking run on vast.ai) into a tidy per-molecule table:
best-pose affinity, number of poses, and affinity spread across poses (a rough
proxy for binding-mode confidence -- tightly clustered top poses suggest a
well-defined pose, widely spread ones suggest an ambiguous docking result).

Each PDBQT file is a docked ligand's multi-model output: MODEL blocks with a
`REMARK VINA RESULT:  <affinity>  <rmsd_lb>  <rmsd_ub>` line each, ordered from
best (most negative kcal/mol) to worst. We only need the affinity numbers here,
not full pose coordinates -- a simple text parse is far faster than loading
every model into RDKit/OpenBabel.
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKED_DIR = PROJECT_ROOT / "tmp" / "docked_all"

AFFINITY_RE = re.compile(r"^REMARK VINA RESULT:\s+(-?\d+\.?\d*)")


def parse_pdbqt_affinities(path: Path) -> list:
    affinities = []
    with open(path) as f:
        for line in f:
            m = AFFINITY_RE.match(line)
            if m:
                affinities.append(float(m.group(1)))
    return affinities


def main() -> None:
    rows = []
    for path in sorted(DOCKED_DIR.glob("*_out.pdbqt")):
        mol_name = path.stem.removesuffix("_out")
        affinities = parse_pdbqt_affinities(path)
        if not affinities:
            continue
        rows.append({
            "Molecule_Name": mol_name,
            "vina_best_affinity": affinities[0],
            "vina_n_poses": len(affinities),
            "vina_affinity_spread": max(affinities) - min(affinities),
            "vina_top3_mean_affinity": sum(affinities[:3]) / len(affinities[:3]),
        })

    df = pd.DataFrame(rows)
    out_path = PROJECT_ROOT / "models" / "cyp3a4_docking_scores.csv"
    df.to_csv(out_path, index=False)
    print(f"Parsed {len(df)} docked ligands, saved to {out_path}")
    print(df.describe())


if __name__ == "__main__":
    main()
