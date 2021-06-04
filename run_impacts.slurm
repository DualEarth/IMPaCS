#!/bin/bash
#SBATCH -p defq
#SBATCH -J impct300
#SBATCH --exclude=gpu01     ### This node is heavily used
#SBATCH --exclude=node01     ### This node is not working
#SBATCH -o impacts-ensemble_300.out
#SBATCH -e impacts-ensemble_300.error
echo 'Running IMPAaCS'${seed}
#python3 -u run_full_impacts.py
python3 -u run_ensemble_impacts.py
