"""Produce the candidate pose selections from the full pool.

Three variants are emitted rather than one, because the two available ranking
signals disagree for 19 of 20 ligands and nothing here can prove which is
right:

  A  cluster   - largest-cluster medoid: the mode of Boltz's own sampled
                 distribution (written by cluster_poses.py).
  B  physics   - lowest GFN2-xTB interaction energy among PoseBusters-valid
                 poses. This is what PXR #2 and #10 did (both top-10). Caveat
                 measured here: the xTB model is ligand + heme + Cys442 only,
                 so it scores the coordination and porphyrin contact but is
                 blind to the rest of the pocket, which biases it toward
                 heme-stacking poses.
  C  fusion    - reciprocal rank fusion of cluster population and xTB dE,
                 mirroring PXR #1, who fused confidence with physics instead
                 of trusting either alone. ipTM is saturated here
                 (0.946-0.990), so cluster population replaces it.

Common to all three: a hard PoseBusters filter (a failing pose scores 0, so it
is never selectable) and, for ligands carrying an accessible aromatic sp2
nitrogen, a preference for Fe-coordinated poses.

xTB dE is comparable only WITHIN a ligand (heme charge/spin assumptions and
ligand composition differ between ligands), which is all that is needed to
pick 1 of N per ligand.

Usage:
    python select_final.py ANALYSIS_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
RDLogger.DisableLog("rdApp.*")
COORDINATING_N = Chem.MolFromSmarts("[n;X2;H0]")
RRF_K = 10


def main(analysis_dir: str) -> None:
    d = Path(analysis_dir)
    feats = pd.read_csv(d / "pose_features.csv")
    pb = pd.read_csv(d / "pb_official.csv")[["pose", "pb_official_pass", "n_failed", "sub_pdb"]]
    df = feats.merge(pb, on="pose", how="left")

    xtb_csv = d / "xtb.csv"
    if xtb_csv.exists():
        x = pd.read_csv(xtb_csv)
        df = df.merge(x.loc[x["xtb_ok"], ["pose", "dE_kcal"]], on="pose", how="left")
    else:
        df["dE_kcal"] = np.nan

    clusters_csv = d / "pose_clusters.csv"
    if clusters_csv.exists():
        cl = pd.read_csv(clusters_csv)[["pose", "cluster"]]
        df = df.merge(cl, on="pose", how="left")
    else:
        df["cluster"] = np.nan

    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    can_coord = {m: bool(Chem.MolFromSmiles(s).GetSubstructMatches(COORDINATING_N))
                 for m, s in smi.items()}

    rows_b, rows_c, report = [], [], []
    for mid, grp in df.groupby("Molecule_Name"):
        pool = grp[grp["pb_official_pass"] == True]  # noqa: E712
        stage = "pb_valid"
        if pool.empty:
            pool, stage = grp.nsmallest(1, "n_failed"), "NO_VALID_POSE"
        elif can_coord.get(mid):
            coord = pool[pool["fe_coordinated"]]
            if not coord.empty:
                pool, stage = coord, "pb_valid+fe_coord"
        pool = pool.reset_index(drop=True)

        # cluster population of each pose (size of the cluster it belongs to)
        if pool["cluster"].notna().any():
            sizes = pool["cluster"].map(pool["cluster"].value_counts())
        else:
            sizes = pd.Series(np.ones(len(pool)), index=pool.index)

        has_e = pool["dE_kcal"].notna().any()
        # B: physics
        pick_b = pool.loc[pool["dE_kcal"].idxmin()] if has_e else pool.iloc[0]
        # C: fusion of cluster population (desc) and dE (asc)
        r_pop = sizes.rank(ascending=False)
        r_de = pool["dE_kcal"].rank(ascending=True) if has_e else pd.Series(
            np.ones(len(pool)), index=pool.index)
        rrf = 1 / (RRF_K + r_pop) + 1 / (RRF_K + r_de)
        pick_c = pool.loc[rrf.idxmax()]

        rows_b.append(pick_b)
        rows_c.append(pick_c)
        report.append(dict(
            Molecule_Name=mid, stage=stage, n_pool=len(pool),
            physics_pose=pick_b["pose"], physics_dE=pick_b["dE_kcal"],
            fusion_pose=pick_c["pose"], fusion_dE=pick_c["dE_kcal"],
            same_bc=pick_b["pose"] == pick_c["pose"],
            fe_coord_b=bool(pick_b["fe_coordinated"]),
            fe_coord_c=bool(pick_c["fe_coordinated"]),
        ))

    for name, rows in (("B_physics", rows_b), ("C_fusion", rows_c)):
        out = pd.DataFrame(rows)
        out.to_csv(d / f"selected_{name}.csv", index=False)
        print(f"wrote selected_{name}.csv ({len(out)} ligands)")

    rep = pd.DataFrame(report)
    rep.to_csv(d / "selection_comparison.csv", index=False)
    with pd.option_context("display.width", 220):
        print(rep.to_string(index=False))
    print(f"\nphysics and fusion agree on {int(rep.same_bc.sum())}/{len(rep)} ligands")


if __name__ == "__main__":
    main(*sys.argv[1:])
