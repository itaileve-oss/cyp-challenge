#!/usr/bin/env bash
# Boltz-2 calibration on the rented GPU: turn every runtime/VRAM estimate into a
# measured number before committing to the full 20-ligand pool.
#
# Measures, on ONE ligand:
#   A  defaults, no bond constraint       -> baseline trunk+1-sample cost
#   B  defaults, with Cys442-FE bond      -> does the constraint work at all
#   C  rcy 5, 10 samples, potentials      -> per-sample marginal cost, peak VRAM
#
# From A and C: t_sample = (t_C - t_A) / 9, which sets the sample budget.
#
# Usage:  bash calibrate_boltz.sh WORKDIR LIGAND_ID
set -euo pipefail

WORK="${1:?usage: calibrate_boltz.sh WORKDIR LIGAND_ID}"
LIG="${2:?usage: calibrate_boltz.sh WORKDIR LIGAND_ID}"
mkdir -p "$WORK"/{yaml,out,log}
cd "$WORK"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee log/gpu.txt

# Peak-VRAM sampler: boltz does not report it, so poll alongside each run.
peak_vram() {
  local tag="$1"; shift
  local vram_log="log/vram_${tag}.txt"
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$vram_log"
      sleep 2
    done ) & local sampler=$!
  local start=$SECONDS
  "$@" > "log/run_${tag}.log" 2>&1 || { kill $sampler; echo "FAILED: $tag"; return 1; }
  local elapsed=$((SECONDS - start))
  kill $sampler 2>/dev/null || true
  local peak=$(sort -n "$vram_log" | tail -1)
  printf '%s\telapsed=%ss\tpeak_vram=%sMiB\n' "$tag" "$elapsed" "$peak" | tee -a log/summary.txt
}

# triton JIT-compiles kernels at runtime and needs a C compiler; the CUDA
# *runtime* base image ships without one, which fails as
# "RuntimeError: Failed to find C compiler".
command -v gcc >/dev/null || { echo "installing gcc for triton"; apt-get update -qq && apt-get install -y -qq build-essential; }

# --- MSA once: same protein for all 20 ligands, so never pay this twice -------
# Reuse boltz's own MERGED msa (boltz_results_*/msa/<id>_0.csv), not the raw
# per-database a3m files it leaves in *_tmp_env/ -- those are server output, not
# the alignment boltz actually consumes.
MSA_CSV="$PWD/cyp3a4_msa.csv"
if [[ ! -f "$MSA_CSV" ]]; then
  echo "== building MSA (once) =="
  python "$SRC/make_boltz_inputs.py" yaml --only "$LIG"
  boltz predict "yaml/${LIG}.yaml" --use_msa_server --out_dir out/msa_seed \
      --diffusion_samples 1 --sampling_steps 20 --output_format pdb \
      > log/msa.log 2>&1 || true
  merged=$(find out/msa_seed -path '*/msa/*_0.csv' | head -1)
  [[ -n "$merged" ]] || { echo "MSA build failed; see log/msa.log"; tail -20 log/msa.log; exit 1; }
  cp "$merged" "$MSA_CSV"
  echo "MSA cached: $(wc -l < "$MSA_CSV") rows from $merged"
fi

# --- A: baseline --------------------------------------------------------------
python "$SRC/make_boltz_inputs.py" yaml_A --only "$LIG" --msa "$MSA_CSV"
peak_vram A boltz predict "yaml_A/${LIG}.yaml" --out_dir out/A --output_format pdb

# --- B: bond constraint -------------------------------------------------------
python "$SRC/make_boltz_inputs.py" yaml_B --only "$LIG" --msa "$MSA_CSV" --bond
peak_vram B boltz predict "yaml_B/${LIG}.yaml" --out_dir out/B --output_format pdb

# --- C: pool-like settings ----------------------------------------------------
peak_vram C boltz predict "yaml_B/${LIG}.yaml" --out_dir out/C --output_format pdb \
    --recycling_steps 5 --diffusion_samples 10 --sampling_steps 200 \
    --step_scale 1.2 --max_parallel_samples 5 --use_potentials

echo "=== summary ==="; cat log/summary.txt
echo "poses written:"; find out -name '*.pdb' | wc -l
