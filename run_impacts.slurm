#!/bin/bash
#SBATCH -p defq
#SBATCH -J impacts
#SBATCH --exclude=gpu01     ### This node is heavily used
#SBATCH --exclude=node01     ### This node is not working
#SBATCH -o impacts.out
#SBATCH -e impacts.error
echo 'Running IMPAaCS'${seed}
python3 -u run_full_impacts.py
