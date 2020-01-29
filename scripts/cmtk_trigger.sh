#!/bin/bash
#SBATCH --job-name=cmtk_t_m
#SBATCH --partition=trc
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./output/slurm-%j.out

cd /oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/20191204_anatomy_collection/20200107_collection/

#f=f21a0.nii
for f in *.nii
do
echo "$f"
cd /home/users/brezovec/projects/bigbadbrain/scripts/
echo $PWD
sbatch cmtk_single.sh $f
done
