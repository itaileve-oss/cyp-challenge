#!/bin/bash
# Dock TEST_BLINDED's compounds against CYP1A2's single receptor (2hi4) --
# just confirmed as a real win (docking+pharmacophore, p=0.0001, 8/8) after
# earlier being scoped out for lack of signal alone. Single receptor, so all
# available workers go to this one job -- much faster than the 5-receptor runs.
set -e
cd /root

N_WORKERS="${N_WORKERS:-64}"

RECEPTOR_NAME="test_cyp1a2_2hi4" RECEPTOR_PDB="receptors/2hi4_prepared.pdb" \
  LIGAND_DIR="test_blinded_all_mol2" HEME_RES_NUM=900 CHAIN=A \
  BOX_X=2.487 BOX_Y=17.834 BOX_Z=20.155 N_WORKERS=$N_WORKERS \
  setsid nohup /opt/conda/bin/python3 /root/run_batch_ensemble.py \
  >> /root/dock_test_cyp1a2_2hi4.log 2>&1 < /dev/null &
disown
echo "launched test_cyp1a2_2hi4 -> pid $!"

sleep 3
echo "--- process check ---"
ps aux | grep run_batch_ensemble | grep -v grep | wc -l
