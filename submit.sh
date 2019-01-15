#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=60:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G

bash motioncorr.sh
