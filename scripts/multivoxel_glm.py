import numpy as np
from time import time
import os
import sys
import scipy

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

import BigBadBrain as bbb

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [25] # 1 index
folders = bbb.get_fly_folders(root_path, desired_flies)
beta_len = 3 #MUST BE ODD

#######################
### Loop over flies ###
#######################

for fly_idx, folder in enumerate(folders):
    
    directory = folder
    announce_start(directory, fly_idx, folders)
    timestamps = load_timestamps(directory)
    fictrac = load_fictrac(directory)
    brain = bbb.get_z_brain(directory, channel='green') 

    ### Create and save multivoxel X matrix ###
    score, betas = bbb.fit_all_voxel_glm(brain, fictrac_interp)
    save_file = os.path.join(directory, 'big_X_single_betas')
    print('Score: {}'.format(score))
    sys.stdout.flush()
    np.save(save_file, betas)
    save_file = os.path.join(directory, 'big_X_single_betas.nii')
    bbb.save_brain(save_file, betas)