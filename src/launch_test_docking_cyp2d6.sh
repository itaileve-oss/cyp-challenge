#!/bin/bash
# Dock TEST_BLINDED's compounds against CYP2D6's 5-receptor ensemble --
# the one isoform (besides CYP3A4) that validated as a real docking win
# (docking+pharma p=0.0137, 7/8).
set -e
cd /root

N_WORKERS="${N_WORKERS:-14}"

launch() {
  local recname=$1 recpdb=$2 heme=$3 chain=$4 bx=$5 by=$6 bz=$7
  RECEPTOR_NAME="test_cyp2d6_${recname}" RECEPTOR_PDB="receptors/${recpdb}" \
    LIGAND_DIR="test_blinded_all_mol2" HEME_RES_NUM=$heme CHAIN=$chain \
    BOX_X=$bx BOX_Y=$by BOX_Z=$bz N_WORKERS=$N_WORKERS \
    setsid nohup /opt/conda/bin/python3 /root/run_batch_ensemble.py \
    >> /root/dock_test_cyp2d6_${recname}.log 2>&1 < /dev/null &
  disown
  echo "launched test_cyp2d6_${recname} -> pid $!"
}

launch 3qm4 3qm4_prepared.pdb 502 A 10.216 -9.535 20.943
launch 3tbg 3tbg_prepared.pdb 800 A 8.347 22.686 -7.611
launch 4wnu 4wnu_prepared.pdb 601 A 3.845 27.984 -6.760
launch 4xrz 4xrz_prepared.pdb 601 A -20.659 28.030 -63.201
launch 5tft 5tft_prepared.pdb 601 A -3.751 28.528 6.335

sleep 3
echo "--- process check ---"
ps aux | grep run_batch_ensemble | grep -v grep | wc -l
echo "processes running (should be 5 main + N_WORKERS*5 total)"
