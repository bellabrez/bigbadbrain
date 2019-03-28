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
from BigBadBrain.glm import fit_glm, save_glm_map
from BigBadBrain.motcorr import get_motcorr_brain, motion_correction

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [25] # 1 index
folders = get_fly_folders(root_path, desired_flies)

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
    brain_file = os.path.join(folder, 'motcorr', 'motcorr_red.nii')
    brain = load_numpy_brain(brain_file)

    ### Perform motion correction ###
    brain_motcorr = motion_correction(brain_master=brain, folder=folder, subfolder='motcorr_on_motcorr')