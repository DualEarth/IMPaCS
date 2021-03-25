#!/bin/bash
#SBATCH -p defq
#SBATCH -J imp-10-45
#SBATCH --exclude=gpu01     ### This node is heavily used
#SBATCH --exclude=node01     ### This node is not working
#SBATCH -o impacts-10m-45lat.out
#SBATCH -e impacts-10m-45lat.error
echo 'Running IMPAaCS'${seed}
python3 -u run_full_impacts.py
