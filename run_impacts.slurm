#!/bin/bash
#SBATCH -p defq
#SBATCH -J impct_unbound
#SBATCH --exclude=gpu01     ### This node is heavily used
#SBATCH --exclude=node01     ### This node is not working
#SBATCH -o impacts-ensemble_60-100_unbound.out
#SBATCH -e impacts-ensemble_60-100_unbound.error
echo 'Running IMPAaCS'
#python3 -u run_full_impacts.py
python3 -u run_ensemble_impacts.py
