#!/bin/bash
# Dock TEST_BLINDED's ~750 compounds (stereo-enumerated) against CYP3A4's
# 5-receptor ensemble, same aligned-frame box center for all 5 (receptors
# were aligned into a common reference frame when the ensemble was built).
# setsid+nohup+disown detachment, same lesson as the CYP1A2/2C9/2D6 run.
#
# Usage: run ON the remote instance: N_WORKERS=<n> bash launch_test_docking.sh

set -e
cd /root

N_WORKERS="${N_WORKERS:-8}"

launch() {
  local recname=$1 recpdb=$2 heme=$3 chain=$4 bx=$5 by=$6 bz=$7
  RECEPTOR_NAME="test_cyp3a4_${recname}" RECEPTOR_PDB="receptors/${recpdb}" \
    LIGAND_DIR="test_blinded_all_mol2" HEME_RES_NUM=$heme CHAIN=$chain \
    BOX_X=$bx BOX_Y=$by BOX_Z=$bz N_WORKERS=$N_WORKERS \
    setsid nohup /opt/conda/bin/python3 /root/run_batch_ensemble.py \
    >> /root/dock_test_cyp3a4_${recname}.log 2>&1 < /dev/null &
  disown
  echo "launched test_cyp3a4_${recname} -> pid $!"
}

# Shared box center (all 5 receptors aligned to a common frame): ref-ligand centroid
BX=-21.94179412
BY=-17.84247059
BZ=-9.83882353

launch 4ny4 cyp3a4_receptor_prepared.pdb 601  A $BX $BY $BZ
launch 1w0f cyp3a4_1w0f_prepared.pdb     1501 A $BX $BY $BZ
launch 1w0g cyp3a4_1w0g_prepared.pdb     1501 A $BX $BY $BZ
launch 3nxu cyp3a4_3nxu_prepared.pdb     508  A $BX $BY $BZ
launch 2j0d cyp3a4_2j0d_prepared.pdb     1497 A $BX $BY $BZ

sleep 3
echo "--- process check ---"
ps aux | grep run_batch_ensemble | grep -v grep | wc -l
echo "processes running (should be 5 main + N_WORKERS*5 total)"
