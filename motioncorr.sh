#! /bin/bash

DD=/scratch/users/brezovec
in=${DD}/TSeries-11162018-1625-002.nii
out=${DD}/motcorr

# average the time series
antsMotionCorr  -d 3 -a $in -o ${out}_avg.nii.gz

# do affine motion correction
antsMotionCorr  -d 3 -o [${out},${out}.nii.gz,${out}_avg.nii.gz] -m gc[ ${out}_avg.nii.gz , $in , 1 , 1 , Random, 0.05  ] -t Affine[ 0.005 ] -i 1 -u 1 -e 1 -s 0 -f 1 -n 100
