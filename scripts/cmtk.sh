#!/bin/bash
#SBATCH --job-name=cmtk
#SBATCH --partition=trc
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output/slurm-%j.out

echo $PWD
cd /home/users/brezovec/temp
echo $PWD

#master="/home/users/brezovec/temp/Meanbrain.nii"
#slave="/home/users/brezovec/temp/Template.nii"

#(cd /home/users/brezovec/temp && exec munger -a -w -X 26 -C 8 -G 80 -R 4 -A '--accuracy 0.4' -W 'accuracy 0.4' -T 4 -s Meanbrain.nii Template.nii)
munger -a -w -X 26 -C 8 -G 80 -R 4 -A '--accuracy 0.4' -W 'accuracy 0.4' -T 4 -s Meanbrain.nii Template.nii