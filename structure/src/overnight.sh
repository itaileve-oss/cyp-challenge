#!/bin/bash
# Unattended pipeline for the full Boltz pool, run ON the vast.ai instance.
#
# Waits for the seeds 2-10 run to finish, then analyses all ~3600 poses on the
# box's 96 cores, ranks them, builds and validates three candidate submission
# zips, and packs everything small into RESULTS.tgz with a READY marker that a
# local watcher polls for.
#
# Seed 1 is deliberately NOT included: it lives only on the laptop, and
# uploading ~1 GB of PDBs to add 10% more poses is not worth the transfer risk.
# Its analysis already exists locally and can be merged at the CSV level.
#
# Everything is idempotent-ish: each stage writes a marker, and a rerun skips
# completed stages, so the script can be restarted after a dropped connection.

set -u
PY=/opt/conda/envs/GNINA/bin/python
SRC=/workspace/cyp/structure/src
RUNS=/workspace/cyp/runs
OUT=/workspace/cyp/analysis/pool
LOG=/workspace/cyp/overnight.log
STAGE=$OUT/stages
mkdir -p "$OUT" "$STAGE"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== overnight pipeline start ==="

# ---------------------------------------------------------------- 1. wait
if [ ! -f "$STAGE/boltz_done" ]; then
  say "waiting for SEEDS2_10_DONE"
  while ! grep -q SEEDS2_10_DONE "$RUNS/progress.txt" 2>/dev/null; do sleep 60; done
  touch "$STAGE/boltz_done"
fi
say "boltz finished: $(grep -c '^seed=' "$RUNS/progress.txt") passes"
say "pose counts per pass:"; grep '^seed=' "$RUNS/progress.txt" | tee -a "$LOG"

SEED_DIRS=$(ls -d "$RUNS"/full_seed*/boltz_results_full_inputs/predictions 2>/dev/null)
TOTAL=$(find $SEED_DIRS -name "*_model_*.pdb" 2>/dev/null | wc -l)
say "total poses found: $TOTAL"

LIGANDS=$($PY -c "
import pandas as pd
print(' '.join(pd.read_csv('/workspace/cyp/data/raw/TEST_BLINDED_structures.csv')['Molecule_Name']))")

# ------------------------------------------- 2. per-ligand analysis, parallel
# Sharding by ligand keeps each job single-threaded and independent; 20 ligands
# across 96 cores means every ligand is analysed concurrently.
analyse_one() {
  mid=$1
  d=/workspace/cyp/analysis/perlig/$mid
  mkdir -p "$d"
  dirs=$(ls -d /workspace/cyp/runs/full_seed*/boltz_results_full_inputs/predictions/$mid 2>/dev/null)
  [ -z "$dirs" ] && { echo "no dirs for $mid"; return 1; }
  # Each step is skipped independently: a failure in the PoseBusters step must
  # not force the expensive geometry analysis to be recomputed on a rerun.
  if [ ! -f "$d/pose_features.csv" ]; then
    OMP_NUM_THREADS=1 /opt/conda/envs/GNINA/bin/python /workspace/cyp/structure/src/analyze_pool.py "$d" $dirs > "$d/analyze.log" 2>&1
  fi
  if [ ! -f "$d/pb_official.csv" ]; then
    OMP_NUM_THREADS=1 /opt/conda/envs/GNINA/bin/python /workspace/cyp/structure/src/pb_official.py "$d" $dirs > "$d/pb.log" 2>&1
  fi
}
export -f analyse_one

if [ ! -f "$STAGE/analysed" ]; then
  say "analysing poses per ligand (20-way parallel)"
  printf '%s\n' $LIGANDS | xargs -P 20 -I{} bash -c 'analyse_one {}'
  # Only mark the stage complete if every ligand actually produced both files;
  # otherwise leave it unmarked so a rerun retries the stragglers.
  nfeat=$(ls /workspace/cyp/analysis/perlig/*/pose_features.csv 2>/dev/null | wc -l)
  npb=$(ls /workspace/cyp/analysis/perlig/*/pb_official.csv 2>/dev/null | wc -l)
  say "per-ligand outputs: pose_features=$nfeat pb_official=$npb (expected 20 each)"
  if [ "$nfeat" -eq 20 ] && [ "$npb" -eq 20 ]; then
    touch "$STAGE/analysed"
  else
    say "ERROR incomplete analysis - inspect analysis/perlig/*/pb.log and rerun"
    for f in /workspace/cyp/analysis/perlig/*/pb.log; do tail -3 "$f" | grep -iE "error|not found|Module" && break; done | tee -a "$LOG"
  fi
fi

# ------------------------------------------------------- 3. merge the shards
say "merging per-ligand CSVs"
$PY - <<'EOF' 2>&1 | tee -a "$LOG"
import pandas as pd
from pathlib import Path
base = Path("/workspace/cyp/analysis/perlig")
out = Path("/workspace/cyp/analysis/pool"); out.mkdir(parents=True, exist_ok=True)
for name in ("pose_features", "pb_official"):
    parts = [pd.read_csv(p) for p in sorted(base.glob(f"*/{name}.csv"))]
    if not parts:
        print(f"NO PARTS for {name}"); continue
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out / f"{name}.csv", index=False)
    print(f"{name}: {len(df)} rows from {len(parts)} ligands")
pb = pd.read_csv(out / "pb_official.csv")
print("PB valid:", int(pb.pb_official_pass.sum()), "/", len(pb))
EOF

# ------------------------------------------------------------- 4. xtb tier 1
if [ ! -f "$STAGE/xtb" ]; then
  XTB=$(ls /opt/conda/envs/GNINA/bin/xtb 2>/dev/null || which xtb 2>/dev/null)
  if [ -n "$XTB" ]; then
    say "xtb tier-1 scoring (90 workers) using $XTB"
    ulimit -s unlimited 2>/dev/null
    $PY "$SRC/xtb_score.py" "$OUT/pb_official.csv" "$OUT/xtb.csv" \
        --valid-only --workers 90 --xtb "$XTB" >> "$LOG" 2>&1
    touch "$STAGE/xtb"
  else
    say "WARNING xtb not installed - skipping tier 1, selection falls back to clustering only"
  fi
fi

# ------------------------------------------------------------ 5. clustering
if [ ! -f "$STAGE/clustered" ]; then
  say "frame-corrected clustering (superpose on CA, symmetry-aware RMSD, 2 A Butina)"
  $PY "$SRC/cluster_poses.py" "$OUT" 2.0 >> "$LOG" 2>&1 && touch "$STAGE/clustered"
fi

# ------------------------------------------------------------- 6. selection
say "building candidate selections"
$PY "$SRC/select_final.py" "$OUT" >> "$LOG" 2>&1

# --------------------------------------------------- 7. zips + validation
cd /workspace/cyp
for v in A_cluster:selected_poses_clustered.csv B_physics:selected_B_physics.csv C_fusion:selected_C_fusion.csv; do
  name=${v%%:*}; csv=${v##*:}
  if [ -f "$OUT/$csv" ]; then
    say "building submission $name"
    $PY "$SRC/build_submission.py" "$OUT/$csv" "/workspace/cyp/submissions/pool_${name}.zip" >> "$LOG" 2>&1
  else
    say "SKIP $name (missing $csv)"
  fi
done

# ------------------------------------------------------------ 8. pack small
say "packing results"
PACK=/workspace/cyp/pack
rm -rf "$PACK"; mkdir -p "$PACK/csv" "$PACK/pdb" "$PACK/zips" "$PACK/logs"
cp "$OUT"/*.csv "$PACK/csv/" 2>/dev/null
cp /workspace/cyp/submissions/*.zip "$PACK/zips/" 2>/dev/null
cp "$RUNS/progress.txt" "$RUNS/vram.log" "$LOG" "$PACK/logs/" 2>/dev/null
# the chosen PDBs, plus the 3 best per ligand by dE for a later tier-2 pass
$PY - <<'EOF' 2>&1 | tee -a "$LOG"
import pandas as pd, shutil
from pathlib import Path
out = Path("/workspace/cyp/analysis/pool"); pack = Path("/workspace/cyp/pack/pdb")
want = set()
for f in out.glob("selected_*.csv"):
    df = pd.read_csv(f)
    if "sub_pdb" in df: want |= set(df["sub_pdb"].dropna())
x = out / "xtb.csv"
if x.exists():
    d = pd.read_csv(x)
    d = d[d.xtb_ok]
    pb = pd.read_csv(out / "pb_official.csv")[["pose", "sub_pdb"]]
    d = d.merge(pb, on="pose", how="left")
    top3 = d.sort_values("dE_kcal").groupby("Molecule_Name").head(3)
    top3.to_csv(out / "top3_by_dE.csv", index=False)
    shutil.copy(out / "top3_by_dE.csv", "/workspace/cyp/pack/csv/")
    want |= set(top3["sub_pdb"].dropna())
n = 0
for p in want:
    p = Path(p)
    if p.exists(): shutil.copy(p, pack / p.name); n += 1
print(f"packed {n} pose PDBs")
EOF
tar czf /workspace/cyp/RESULTS.tgz -C /workspace/cyp pack
du -sh /workspace/cyp/RESULTS.tgz | tee -a "$LOG"
echo READY > /workspace/cyp/RESULTS_READY
say "=== DONE - RESULTS.tgz ready ==="
