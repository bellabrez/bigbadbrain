#!/bin/bash
#SBATCH --job-name=cmtk_s_m
#SBATCH --partition=trc
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=./output/slurm-%j.out

#cd /oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/20191204_anatomy_collection/20200107_collection/
cd /oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/20200121_warps/

#master=diegomean.nii
master=diegoflip.nii
#slave=kevinmean.nii
#master=kevintemplate.nii
slave=meanbrain20200113.nii

savefolder=out

#slave="$1"
#friend1=PB_L.nii.gz
#friend2=PB_R.nii.gz

echo "master" $master
echo "slave" $slave

munger -a -w -X 26 -C 8 -G 80 -R 4 -A '--accuracy 0.4' -W '--accuracy 0.4' -T 8 -s $master $slave
cmtk reformatx -o ./${savefolder}/${slave::-4}_2_${master::-4}.nii --floating ${slave} ${master} Registration/warp/${master::-4}_${slave}_warp_m0g80c8e1e-1x26r4.list/

#cmtk reformatx --nn -o ./${savefolder}/${friend1::-7}_warped.nii --floating ${friend1} ${master} Registration/warp/${master::-4}_${slave}_warp_m0g80c8e1e-1x26r4.list/
#cmtk reformatx --nn -o ./${savefolder}/${friend2::-7}_warped.nii --floating ${friend2} ${master} Registration/warp/${master::-4}_${slave}_warp_m0g80c8e1e-1x26r4.list/
