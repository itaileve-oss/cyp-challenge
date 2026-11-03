#!/bin/bash
# Deploy GalaxyDock2-HEME + Chimera + CYP1A2/CYP2C9/CYP2D6's 11 receptors +
# the already-prepared TEST_BLINDED ligand set (reused as-is -- same 750
# molecules regardless of which isoform's receptors they're docked against)
# to a fresh vast.ai instance. All known runtime deps installed up front.
#
# Usage: SSH_PORT=xxxxx SSH_HOST=x.x.x.x bash deploy_test_docking_other_isoforms.sh

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

echo "=== 3. Upload GalaxyDock2-HEME + Chimera ==="
$SCP $PROJ/external/GalaxyDock2_HEME.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root && tar xzf GalaxyDock2_HEME.tar.gz -C /root/project/ 2>&1 | tail -3; ls /root/project/"
tar czf /tmp/chimera.tar.gz -C /home/ileven/tools UCSF-Chimera64
$SCP /tmp/chimera.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root && tar xzf chimera.tar.gz -C /root/tools/ 2>&1 | tail -3"

echo "=== 4. Upload CYP1A2/CYP2C9/CYP2D6 receptors (11 total) ==="
$SCP $PROJ/data/receptors/cyp1a2/*_prepared.pdb root@$SSH_HOST:/root/project/receptors/
$SCP $PROJ/data/receptors/cyp2c9/*_prepared.pdb root@$SSH_HOST:/root/project/receptors/
$SCP $PROJ/data/receptors/cyp2d6/*_prepared.pdb root@$SSH_HOST:/root/project/receptors/

echo "=== 5. Upload TEST_BLINDED ligand set (reused from CYP3A4 run) ==="
tar czf /tmp/test_blinded_ligands_full.tar.gz -C $PROJ/data/structures test_blinded_all_mol2
$SCP /tmp/test_blinded_ligands_full.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root/project && tar xzf /root/test_blinded_ligands_full.tar.gz"

echo "=== 6. Upload run_batch_ensemble.py ==="
$SCP $PROJ/src/run_batch_ensemble.py root@$SSH_HOST:/root/

echo "=== Verify ==="
$SSH "ls /root/project/receptors/ | wc -l; ls /root/project/ | grep test_blinded; nproc"

echo "=== Deploy done. Waiting on validation results before launching. ==="
