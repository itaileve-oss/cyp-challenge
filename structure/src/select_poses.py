"""Pick one pose per ligand from the Boltz-2 pool.

Selection, in order:

1. HARD FILTER — the pose must pass the organizers' PoseBusters check.
   `POSEBUSTERS_MAX_FAILURES = 0`, so a single failed check zeroes LDDT-PLI;
   an invalid pose is worth exactly nothing and is never selected.
2. CHEMISTRY PRIOR — a ligand carrying an accessible aromatic sp2 nitrogen
   (azole / pyridine / triazole, no H on N) is a candidate type-II binder that
   coordinates the heme iron. For those ligands only, poses showing N-Fe
   coordination are preferred; the prior is never applied to ligands that have
   no such nitrogen, where a carbon nearest the iron is the correct answer.
3. RANK FUSION — reciprocal rank fusion of ipTM and a consensus score, rather
   than ipTM alone, which is known to be fooled by unphysical poses. The
   consensus score is the negative mean RMSD from a pose to every other
   surviving pose: the pool's own crowded region, i.e. the mode of the
   sampler's distribution, which is a better LDDT-PLI bet than one outlier
   with a high confidence number.

Usage:
    python select_poses.py ANALYSIS_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
RDLogger.DisableLog("rdApp.*")

# Aromatic sp2 N with no hydrogen: an available lone pair for Fe coordination.
# Excludes pyrrole/indole NH and tertiary aromatic N, neither of which can bind.
COORDINATING_N = Chem.MolFromSmarts("[n;X2;H0]")
RRF_K = 10  # reciprocal-rank-fusion damping


def rmsd_matrix(sdfs: list[str]) -> np.ndarray:
    """Direct pairwise RMSD: all samples share one receptor frame, so no
    superposition is needed (and none is wanted - it would hide translation)."""
    coords = []
    for s in sdfs:
        m = Chem.MolFromMolFile(s, sanitize=False)
        coords.append(None if m is None else m.GetConformer().GetPositions())
    n = len(coords)
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if coords[i] is not None and coords[j] is not None:
                out[i, j] = np.sqrt(((coords[i] - coords[j]) ** 2).sum(axis=1).mean())
    return out


def main(analysis_dir: str) -> None:
    d = Path(analysis_dir)
    feats = pd.read_csv(d / "pose_features.csv")
    pb = pd.read_csv(d / "pb_official.csv")
    df = feats.merge(pb[["pose", "pb_official_pass", "n_failed", "failed", "sub_pdb"]],
                     on="pose", how="left")

    smi = pd.read_csv(ROOT / "data/raw/TEST_BLINDED_structures.csv").set_index("Molecule_Name")["SMILES"]
    type_ii = {m: bool(Chem.MolFromSmiles(s).GetSubstructMatches(COORDINATING_N))
               for m, s in smi.items()}
    df["ligand_can_coordinate"] = df["Molecule_Name"].map(type_ii)

    picks, report = [], []
    for mid, grp in df.groupby("Molecule_Name"):
        stage = "pb_valid"
        pool = grp[grp["pb_official_pass"] == True]  # noqa: E712
        if pool.empty:
            # Nothing valid: take the fewest failed checks, score will be 0 but
            # a pose must still be submitted for every ligand.
            pool, stage = grp.nsmallest(1, "n_failed"), "NO_VALID_POSE"

        if type_ii[mid] and stage == "pb_valid":
            coord = pool[pool["fe_coordinated"]]
            if not coord.empty:
                pool, stage = coord, "pb_valid+fe_coord"

        pool = pool.reset_index(drop=True)
        if len(pool) > 1:
            m = rmsd_matrix(pool["lig_sdf"].tolist())
            np.fill_diagonal(m, np.nan)
            consensus = -np.nanmean(m, axis=1)
            r_conf = pd.Series(pool["ligand_iptm"]).rank(ascending=False)
            r_cons = pd.Series(consensus).rank(ascending=False)
            pool = pool.assign(
                consensus_mean_rmsd=-consensus,
                rrf=1 / (RRF_K + r_conf) + 1 / (RRF_K + r_cons),
            )
            best = pool.loc[pool["rrf"].idxmax()]
        else:
            best = pool.assign(consensus_mean_rmsd=np.nan, rrf=np.nan).iloc[0]

        picks.append(best)
        report.append(dict(
            Molecule_Name=mid, pose=best["pose"], stage=stage,
            n_pb_valid=int((grp["pb_official_pass"] == True).sum()),  # noqa: E712
            can_coordinate=type_ii[mid], fe_coordinated=best["fe_coordinated"],
            min_fe_dist=round(best["min_fe_dist"], 2), fe_atom=best["fe_atom"],
            distal=best["distal"], n_pocket_contacts=int(best["n_pocket_contacts"]),
            ligand_iptm=round(best["ligand_iptm"], 3),
            consensus_mean_rmsd=round(best["consensus_mean_rmsd"], 2)
            if pd.notna(best["consensus_mean_rmsd"]) else np.nan,
            sub_pdb=best["sub_pdb"],
        ))

    rep = pd.DataFrame(report)
    pd.DataFrame(picks).to_csv(d / "selected_poses_full.csv", index=False)
    rep.to_csv(d / "selected_poses.csv", index=False)
    with pd.option_context("display.width", 250):
        print(rep.drop(columns="sub_pdb").to_string(index=False))
    print(f"\nstages: {dict(rep.stage.value_counts())}")
    if (rep.stage == "NO_VALID_POSE").any():
        print("WARNING no valid pose (scores 0):",
              ", ".join(rep.loc[rep.stage == "NO_VALID_POSE", "Molecule_Name"]))


if __name__ == "__main__":
    main(*sys.argv[1:])
