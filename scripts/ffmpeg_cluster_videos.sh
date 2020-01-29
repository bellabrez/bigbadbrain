#!/bin/bash
#SBATCH --job-name=ffmpeg
#SBATCH --partition=trc
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=./output/slurm-%j.out

ml system
ml ffmpeg/4.2.1

dir="/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/figs/20191227_cluster_videos"
save_dir="/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/figs/20191227_videos_output"

echo "dir" ${dir}

for cluster in ${dir}/*/
do
#echo "clusster" ${cluster}
echo ${cluster}
#echo ${cluster::-3}
num_temp=${cluster: -3}
num=${num_temp::-1}
echo $num
ffmpeg -framerate 20 -i ${cluster}%*.png -vcodec mpeg4 -b 5000k -vf scale=2000:-2 -y ${save_dir}/cluster_${num}.mp4
done

#munger -a -w -X 26 -C 8 -G 80 -R 4 -A '--accuracy 0.4' -W '--accuracy 0.4' -T 4 -s $master $slave
#cmtk reformatx -o ./${savefolder}/${slave::-4}_warped.nii --floating ${slave} ${master} Registration/warp/${master::-4}_${slave}_warp_m0g80c8e1e-1x26r4.list/
#cmtk reformatx --nn -o ./${savefolder}/${friend1::-7}_warped.nii --floating ${friend1} ${master} Registration/warp/${master::-4}_${slave}_warp_m0g80c8e1e-1x26r4.list/
#cmtk reformatx --nn -o ./${savefolder}/${friend2::-7}_warped.nii --floating ${friend2} ${master} Registration/warp/${master::-4}_${slave}_warp_m0g80c8e1e-1x26r4.list/
