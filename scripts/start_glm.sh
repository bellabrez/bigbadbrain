#!/bin/bash
#SBATCH --job-name=glm
#SBATCH --partition=trc
#SBATCH --mail-user=brezovec@stanford.edu
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="12"

module load gcc/6.3.0
module load python/3.6.1
module load py-numpy/1.14.3_py36
module load py-pandas/0.23.0_py36
module load viz
module load py-scikit-learn/0.19.1_py36

python3 /home/users/brezovec/projects/bigbadbrain/scripts/glm_master_20190617-0618.py
