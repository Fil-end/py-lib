#!/bin/sh
#BSUB -n 24     
#BSUB -R "span[hosts=1]"  ### ask for 1 node
#BSUB -q normal
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J lasp
#BSUB -m node01

#export OMP_NUM_THREADS=2
cd $LS_SUBCWD
NP=`echo $LSB_HOSTS | wc -w`
source /opt/intelstart.sh
mpirun -np $NP /data/apps/lasp/lasp 

python occupation.py

