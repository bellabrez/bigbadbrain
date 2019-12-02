#!/bin/bash
#SBATCH --job-name=cmtk
#SBATCH --partition=trc
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output/slurm-%j.out

master="/home/users/brezovec/temp/Meanbrain.nii"
slave="/home/users/brezovec/temp/Template.nii"

munger -a -w -X 26 -C 8 -G 80 -R 4 -A '--accuracy 0.4' -W 'accuracy 0.4' -T 4 -s "$master" "$slave"