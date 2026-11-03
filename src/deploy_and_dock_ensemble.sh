#!/bin/bash
# Upload GalaxyDock2-HEME + Chimera + receptors + ligands for CYP1A2/CYP2C9/CYP2D6
# to a vast.ai instance, then launch docking for every (isoform, receptor) pair,
# each detached (nohup+disown) so a dropped SSH connection can't kill them --
# same lesson learned the hard way on the CYP3A4 ensemble run.
#
# Usage: SSH_PORT=xxxxx SSH_HOST=x.x.x.x bash deploy_and_dock_ensemble.sh

set -e
SSH_PORT="${SSH_PORT:?set SSH_PORT}"
SSH_HOST="${SSH_HOST:?set SSH_HOST}"
SSH="ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST"
SCP="scp -P $SSH_PORT -o StrictHostKeyChecking=no"
PROJ=/home/ileven/Projects/cyp-challenge

echo "=== 1. Remote dirs ==="
$SSH "mkdir -p /root/project /root/tools"

echo "=== 2. Upload GalaxyDock2-HEME + Chimera (large, one-time) ==="
$SCP $PROJ/external/GalaxyDock2_HEME.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root && tar xzf GalaxyDock2_HEME.tar.gz -C /root/project/ 2>&1 | tail -3; ls /root/project/"
tar czf /tmp/chimera.tar.gz -C /home/ileven/tools UCSF-Chimera64
$SCP /tmp/chimera.tar.gz root@$SSH_HOST:/root/
$SSH "cd /root && tar xzf chimera.tar.gz -C /root/tools/ 2>&1 | tail -3"

echo "=== 3. Upload receptors ==="
$SSH "mkdir -p /root/project/receptors"
$SCP $PROJ/data/receptors/cyp1a2/*_prepared.pdb $PROJ/data/receptors/cyp1a2/*_prepared.box.txt root@$SSH_HOST:/root/project/receptors/ 2>/dev/null || true
$SCP $PROJ/data/receptors/cyp2c9/*_prepared.pdb $PROJ/data/receptors/cyp2c9/*_prepared.box.txt root@$SSH_HOST:/root/project/receptors/
$SCP $PROJ/data/receptors/cyp2d6/*_prepared.pdb $PROJ/data/receptors/cyp2d6/*_prepared.box.txt root@$SSH_HOST:/root/project/receptors/

echo "=== 4. Upload ligand sets (tar first, big directories) ==="
for iso in cyp1a2 cyp2c9 cyp2d6; do
  tar czf /tmp/${iso}_ligands.tar.gz -C $PROJ/data/structures ${iso}_all_mol2_v2
  $SCP /tmp/${iso}_ligands.tar.gz root@$SSH_HOST:/root/
  $SSH "cd /root/project && tar xzf /root/${iso}_ligands.tar.gz"
done

echo "=== 5. Upload run_batch_ensemble.py ==="
$SCP $PROJ/src/run_batch_ensemble.py root@$SSH_HOST:/root/

echo "=== All files uploaded. Verify on remote: ==="
$SSH "ls /root/project/receptors/; echo ---; ls /root/project/ | grep mol2_v2; echo ---; nproc"

echo "=== Deploy done. Launch docking separately with launch_docking_jobs.sh ==="
