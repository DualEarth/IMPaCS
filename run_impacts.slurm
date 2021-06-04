#!/bin/bash
#SBATCH -p defq
#SBATCH -J impacts1-9
#SBATCH --exclude=gpu01     ### This node is heavily used
#SBATCH --exclude=node01     ### This node is not working
#SBATCH -o impacts-ensemble_1-9.out
#SBATCH -e impacts-ensemble_1-9.error
echo 'Running IMPAaCS'${seed}
#python3 -u run_full_impacts.py
python3 -u run_ensemble_impacts.py
