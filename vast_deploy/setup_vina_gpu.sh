#!/bin/bash
# Build Vina-GPU-2.1 on a vast.ai NVIDIA GPU instance.
# Assumes: Ubuntu, NVIDIA GPU + driver already present (standard on vast.ai
# CUDA-image templates), root or sudo access, internet access.
#
# Usage: bash setup_vina_gpu.sh
set -euo pipefail

apt-get update
apt-get install -y --no-install-recommends \
    build-essential libboost-all-dev ocl-icd-opencl-dev opencl-headers \
    nvidia-cuda-toolkit clinfo git wget

echo "=== clinfo (checking OpenCL is visible) ==="
clinfo | head -20 || echo "WARNING: clinfo found no OpenCL platform -- GPU driver/OpenCL may not be set up"

echo "=== nvidia-smi ==="
nvidia-smi || echo "WARNING: nvidia-smi failed"

cd /root
git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git
cd Vina-GPU-2.1

# Find the actual OpenCL lib path (varies by system -- usually alongside the
# NVIDIA driver, not the CUDA toolkit itself)
OPENCL_LIB=$(find / -name "libOpenCL.so*" 2>/dev/null | head -1 | xargs dirname)
echo "Detected OpenCL lib path: $OPENCL_LIB"

# There should be a subdirectory per method, e.g. QuickVina2-GPU-2.1,
# AutoDock-Vina-GPU-2.1 -- adjust METHOD_DIR below if the clone layout differs.
ls -la
echo "=== Pick a METHOD_DIR from the listing above (e.g. QuickVina2-GPU-2.1) and edit its Makefile: ==="
echo "    BOOST_LIB_PATH := $(find / -maxdepth 6 -type d -iname 'boost*' 2>/dev/null | head -1)"
echo "    OPENCL_LIB_PATH := ${OPENCL_LIB:-/usr/lib/x86_64-linux-gnu}"
echo "    GPU_PLATFORM := -DNVIDIA_PLATFORM"
echo "    OPENCL_VERSION := -DOPENCL_3_0"
echo ""
echo "Then, inside that method's directory, run:"
echo "    ulimit -s 8192"
echo "    make clean && make source   # first build, compiles OpenCL kernels from source"
echo "    make clean && make          # rebuild using cached kernel binary (faster)"
