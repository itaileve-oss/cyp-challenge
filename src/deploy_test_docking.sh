#!/bin/bash
# Deploy GalaxyDock2-HEME + Chimera + CYP3A4's 5-receptor ensemble + the
# 750-compound TEST_BLINDED ligand set to a fresh vast.ai instance, then
# install every runtime dependency this session already found missing on a
# fresh image (numpy/scipy + the libGL/libXft/libXss/etc chain) up front in
# one shot, instead of the iterative smoke-test-then-fix cycle from the
# CYP1A2/2C9/2D6 run.
#
# Usage: SSH_PORT=xxxxx SSH_HOST=x.x.x.x bash deploy_test_docking.sh

set -e
SSH_PORT="${SSH_PORT:?set SSH_PORT}"
SSH_HOST="${SSH_HOST:?set SSH_HOST}"
SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST"
SCP="scp -P $SSH_PORT -o StrictHostKeyChecking=no"
PROJ=/home/ileven/Projects/cyp-challenge

echo "=== 1. Remote dirs ==="
$SSH "mkdir -p /root/project /root/tools /root/project/receptors"

echo "=== 2. Install all known runtime deps up front ==="
$SSH "
  apt-get update -qq
  apt-get install -y -qq libfreetype6 libgl1 libsm6 libxt6 libxrender1 libxext6 \
    libxft2 libxss1 libxmu6 libxpm4 libxcursor1 libxinerama1 libxrandr2 libxi6 2>&1 | tail -5
  /opt/conda/bin/pip install -q numpy scipy
  nproc
"

echo "=== 3. Upload GalaxyDock2-HEME + Chimera (large, one-time) ==="
$SCP $PROJ/external/GalaxyDock2_HEME.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root && tar xzf GalaxyDock2_HEME.tar.gz -C /root/project/ 2>&1 | tail -3; ls /root/project/"
tar czf /tmp/chimera.tar.gz -C /home/ileven/tools UCSF-Chimera64
$SCP /tmp/chimera.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root && tar xzf chimera.tar.gz -C /root/tools/ 2>&1 | tail -3"

echo "=== 4. Upload CYP3A4's 5 receptors ==="
$SCP $PROJ/data/receptors/cyp3a4_receptor_prepared.pdb root@$SSH_HOST:/root/project/receptors/
$SCP $PROJ/data/receptors/ensemble/cyp3a4_1w0f_prepared.pdb \
     $PROJ/data/receptors/ensemble/cyp3a4_1w0g_prepared.pdb \
     $PROJ/data/receptors/ensemble/cyp3a4_3nxu_prepared.pdb \
     $PROJ/data/receptors/ensemble/cyp3a4_2j0d_prepared.pdb \
     root@$SSH_HOST:/root/project/receptors/

echo "=== 5. Upload TEST_BLINDED ligand set ==="
tar czf /tmp/test_blinded_ligands.tar.gz -C $PROJ/data/structures test_blinded_all_mol2
$SCP /tmp/test_blinded_ligands.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root/project && tar xzf /root/test_blinded_ligands.tar.gz"

echo "=== 6. Upload run_batch_ensemble.py ==="
$SCP $PROJ/src/run_batch_ensemble.py root@$SSH_HOST:/root/

echo "=== Verify ==="
$SSH "ls /root/project/receptors/; echo ---; ls /root/project/ | grep test_blinded; echo ---; nproc"

echo "=== Deploy done. Run a smoke test before launching the full batch. ==="
