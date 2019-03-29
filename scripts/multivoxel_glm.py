import numpy as np
from time import time
import os
import sys
import scipy

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

from BigBadBrain.brain import bleaching_correction, z_score_brain, get_resolution, save_brain, load_numpy_brain, get_dims
from BigBadBrain.fictrac import load_fictrac, prep_fictrac
from BigBadBrain.utils import load_timestamps, get_fly_folders, send_email
from BigBadBrain.glm import fit_glm, save_glm_map, create_multivoxel_X_matrix
from BigBadBrain.motcorr import get_motcorr_brain, motion_correction

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [25] # 1 index
folders = get_fly_folders(root_path, desired_flies)
beta_len = 3 #MUST BE ODD

#######################
### Loop over flies ###
#######################

for fly_idx, folder in enumerate(folders):
    
    ### Send email and define folder path ###
    function_durations = []
    print('\n ~~~~ Starting analysis of {} ~~~~'.format(folder))
    sys.stdout.flush()
    send_email('Starting {} ({} of {}).'.format(folder, fly_idx+1, len(folders)), 'wow')


    ### Load brain ###
    print('\n~~ Loading Brain ~~')
    sys.stdout.flush()
    z_brain_file = os.path.join(folder, 'brain_zscored_green.nii')
    try:
        brain = load_numpy_brain(z_brain_file)
        dims = get_dims(brain)
    except:
        brain_file = os.path.join(folder, 'motcorr', 'motcorr_green.nii')
        brain = load_numpy_brain(brain_file)
        dims = get_dims(brain)

        ### Bleaching correction (per voxel) ###
        brain = bleaching_correction(brain)

        ### Z-score brain ###
        brain = z_score_brain(brain)
        save_brain(zbrain_file, brain)

    ### Create and save multivoxel X matrix ###
    X = create_multivoxel_X_matrix(brain, dims, beta_len)
    save_file = os.path.join(folder, 'big_X')
    np.save(save_file, X)
