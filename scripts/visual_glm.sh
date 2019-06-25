#!/bin/bash
#SBATCH --job-name=visual_glm
#SBATCH --partition=trc
#SBATCH --time=3:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=./output/slurm-%j.out


ml gcc/6.3.0
ml python/3.6.1
ml py-numpy/1.14.3_py36
ml py-pandas/0.23.0_py36
ml viz
ml py-scikit-learn/0.19.1_py36

EXPT="$1"
CHANNEL="$2"
STIM_IDX="$3"

python3 /home/users/brezovec/projects/bigbadbrain/scripts/visual_glm.py "$EXPT" "$CHANNEL" "$STIM_IDX"