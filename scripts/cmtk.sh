#!/bin/bash
#SBATCH --job-name=cmtk
#SBATCH --partition=trc
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output/slurm-%j.out
#SBATCH --mail-type=ALL

echo $PWD
cd /home/users/brezovec/temp
echo $PWD

master=Meanbrain.nii
slave=Template.nii

munger -a -w -X 26 -C 8 -G 80 -R 4 -A '--accuracy 0.4' -W '--accuracy 0.4' -T 4 -s $master $slave
cmtk reformatx -o ./warped/${master}_warped.nii --floating Template.nii Meanbrain.nii Registration/warp/Meanbrain_Template.nii_warp_m0g80c8e1e-1x26r4.list/