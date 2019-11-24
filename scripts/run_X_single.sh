#!/bin/bash
#SBATCH --job-name=run_X
#SBATCH --partition=trc
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output/slurm-%j.out
#SBATCH --mail-type=ALL

ml gcc/6.3.0
ml python/3.6.1
ml py-numpy/1.14.3_py36
ml py-pandas/0.23.0_py36
ml viz
ml py-scikit-learn/0.19.1_py36

VARS="$1"
echo "$VARS"
python3 /home/users/brezovec/projects/bigbadbrain/bigbadbrain/anatomy_warp.py $VARS
