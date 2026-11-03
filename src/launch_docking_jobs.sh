#!/bin/bash
# Launch GalaxyDock2-HEME docking for all 11 (isoform, receptor) combinations,
# each detached (setsid + nohup + disown) so a dropped SSH connection can't
# kill them -- the exact failure mode that killed the CYP3A4 ensemble run's
# first attempt.
#
# Usage (run ON the remote instance, or via ssh -p PORT root@HOST 'bash -s' < this file):

set -e
cd /root

# 80 cores on this instance / 11 concurrent receptor jobs ~= 7.3 -- unlike the
# CYP3A4 run (128 cores / 5 receptors, default N_WORKERS=16 was fine), the
# default 16 workers * 11 receptors = 176 would badly oversubscribe here.
N_WORKERS=7

launch() {
  local iso=$1 recname=$2 recpdb=$3 heme=$4 chain=$5 bx=$6 by=$7 bz=$8
  RECEPTOR_NAME="${iso}_${recname}" RECEPTOR_PDB="receptors/${recpdb}" \
    LIGAND_DIR="${iso}_all_mol2_v2" HEME_RES_NUM=$heme CHAIN=$chain \
    BOX_X=$bx BOX_Y=$by BOX_Z=$bz N_WORKERS=$N_WORKERS \
    setsid nohup /opt/conda/bin/python3 /root/run_batch_ensemble.py \
    >> /root/dock_${iso}_${recname}.log 2>&1 < /dev/null &
  disown
  echo "launched ${iso}_${recname} -> pid $!"
}

# CYP1A2 (single receptor, only 1 real structure available)
launch cyp1a2 2hi4 2hi4_prepared.pdb 900 A 2.487 17.834 20.155

# CYP2C9 (5 receptors)
launch cyp2c9 1r9o 1r9o_prepared.pdb 500 A 8.636 33.142 -1.903
launch cyp2c9 4nz2 4nz2_prepared.pdb 501 A -40.028 -65.180 -27.305
launch cyp2c9 5a5i 5a5i_prepared.pdb 1494 A 71.143 1.592 -1.779
launch cyp2c9 5k7k 5k7k_prepared.pdb 501 A 24.493 -29.806 2.034
launch cyp2c9 5w0c 5w0c_prepared.pdb 500 A -35.815 61.358 5.716

# CYP2D6 (5 receptors)
launch cyp2d6 3qm4 3qm4_prepared.pdb 502 A 10.216 -9.535 20.943
launch cyp2d6 3tbg 3tbg_prepared.pdb 800 A 8.347 22.686 -7.611
launch cyp2d6 4wnu 4wnu_prepared.pdb 601 A 3.845 27.984 -6.760
launch cyp2d6 4xrz 4xrz_prepared.pdb 601 A -20.659 28.030 -63.201
launch cyp2d6 5tft 5tft_prepared.pdb 601 A -3.751 28.528 6.335

sleep 3
echo "--- process check ---"
ps aux | grep run_batch_ensemble | grep -v grep | wc -l
echo "processes running (should be 11 main + N_WORKERS*11 total)"
