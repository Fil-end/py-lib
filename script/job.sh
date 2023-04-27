#!/bin/bash
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpu
#BSUB -n 1
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -J PPOtest
#BSUB -m node02


source activate
conda activate name
python PPO.py

