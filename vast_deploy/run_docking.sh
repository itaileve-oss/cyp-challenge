#!/bin/bash
# Run the built Vina-GPU-2.1 binary against the full CYP3A4 ligand batch.
# Run from inside the built method directory (e.g. Vina-GPU-2.1/QuickVina2-GPU-2.1/),
# after `make source` has succeeded.
#
# Usage: bash run_docking.sh <path-to-binary> <receptor.pdbqt> <ligand_dir> <output_dir>
set -euo pipefail

BINARY="${1:?path to built Vina-GPU binary}"
RECEPTOR="${2:?path to receptor .pdbqt}"
LIGAND_DIR="${3:?path to ligand pdbqt directory}"
OUTPUT_DIR="${4:?output directory}"

mkdir -p "$OUTPUT_DIR"
ulimit -s 8192

# Box computed from the CYP3A4 reference ligand (2QH, PDB 4NY4) + 8 A padding,
# same box used for the earlier CPU GNINA pilot run.
"$BINARY" \
    --receptor "$RECEPTOR" \
    --ligand_directory "$LIGAND_DIR" \
    --output_directory "$OUTPUT_DIR" \
    --center_x -22.249 --center_y -18.066 --center_z -9.219 \
    --size_x 24.310 --size_y 28.841 --size_z 25.309 \
    --thread 8000
