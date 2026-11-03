#!/usr/bin/env bash
# Generate the Boltz-2 pose pool for all 20 CYP3A4 structure-track ligands.
#
# Settings are the ones the 2026-10-27 ablation selected, NOT the ones intuition
# suggested (see memory/project_cyp_structure_track.md):
#   * plain defaults win: NO --use_potentials (it destroys heme-Fe coordination,
#     100% -> 0%), NO bond constraint, step_scale left at its 1.5 default
#     (lowering it bought +4% diversity for a validity collapse).
#   * sampling_steps 100 is free: same validity and Fe coordination as 200, 2x
#     the poses per unit time.
#   * cost scales as diffusion_samples x sampling_steps at ~8.3 sample-steps/s;
#   * max_parallel_samples MUST be set (default None = no cap -> silent OOM).
#
# Resumable: boltz reuses cached predictions, so re-running skips finished
# ligands. Pass --override to force a rebuild.
#
# Usage: bash run_production_pool.sh WORKDIR MSA_CSV [SAMPLES] [SEED]
set -euo pipefail

WORK="${1:?usage: run_production_pool.sh WORKDIR MSA_CSV [SAMPLES]}"
MSA="${2:?need the cached merged MSA csv}"
SAMPLES="${3:-100}"
SEED="${4:-0}"

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORK"/{yaml,log}
cd "$WORK"

command -v gcc >/dev/null || { apt-get update -qq && apt-get install -y -qq build-essential; }

# One YAML per ligand: protein + HEM + verbatim challenge SMILES, no constraints.
python "$SRC/make_boltz_inputs.py" yaml --msa "$MSA"
n_yaml=$(find yaml -name '*.yaml' | wc -l)
[[ "$n_yaml" -eq 20 ]] || { echo "expected 20 YAMLs, got $n_yaml"; exit 1; }

echo "=== pool: $SAMPLES samples x 20 ligands, seed $SEED, started $(date -u +%FT%TZ) ==="
awk -v s="$SAMPLES" 'BEGIN{printf "est. %.0f min/ligand, %.1f h total (at 8.5 sample-steps/s)\n", s*100/8.5/60, s*100*20/8.5/3600}'

# Peak-VRAM sampler for the whole run.
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> log/vram.txt; sleep 30; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

T0=$SECONDS
# Directory input => boltz batches all 20 and skips any already predicted.
#
# --max_parallel_samples MUST be set: its default is None, meaning every
# diffusion sample is allocated at once. At 200 samples that is a single 16.5 GB
# allocation which OOMs, and boltz then prints only
# "WARNING: ran out of memory, skipping batch" and writes NOTHING while
# still exiting 0. Capping at 20 held 60 samples to 27.3 GB.
boltz predict yaml \
    --out_dir out \
    --output_format pdb \
    --recycling_steps 5 \
    --diffusion_samples "$SAMPLES" \
    --sampling_steps 100 \
    --max_parallel_samples 20 \
    --seed "$SEED" \
    > log/pool.log 2>&1
ELAPSED=$((SECONDS - T0))

n_poses=$(find out -name '*_model_*.pdb' | wc -l)
expected=$((SAMPLES * 20))
echo "=== done in $((ELAPSED / 60)) min, peak VRAM $(sort -n log/vram.txt | tail -1) MiB ==="
echo "poses: $n_poses (expected $expected)"
find out -name '*_model_*.pdb' | sed 's|.*/predictions/||; s|/.*||' | sort | uniq -c | sort -k2

# Fail loudly on the silent-OOM-skip path rather than reporting success.
n_oom=$(grep -ci "ran out of memory" log/pool.log || true)
if [[ "$n_oom" -gt 0 ]]; then
  echo "ERROR: $n_oom OOM batch-skips — lower --diffusion_samples or --max_parallel_samples"
fi
if [[ "$n_poses" -lt "$expected" ]]; then
  echo "ERROR: only $n_poses/$expected poses written"
  exit 1
fi
echo "ALL_POOL_DONE"
