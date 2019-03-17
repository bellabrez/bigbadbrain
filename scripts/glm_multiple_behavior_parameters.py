import numpy as np
from time import time
import os
import sys
import scipy

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

from bigbrain.brain import bleaching_correction
from bigbrain.brain import z_score_brain
from bigbrain.brain import get_resolution
from bigbrain.brain import save_brain
from bigbrain.fictrac import load_fictrac
from bigbrain.fictrac import prep_fictrac
from bigbrain.utils import load_timestamps
from bigbrain.utils import get_fly_folders
from bigbrain.utils import send_email
from bigbrain.glm import fit_glm
from bigbrain.glm import save_glm_map
from bigbrain.motcorr import get_motcorr_brain

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [11,12,13,14,17,19,20] # 1 index
folders = get_fly_folders(root_path, desired_flies)

beta_len = 21 #MUST BE ODD
fps = 50 #of fictrac camera
dur = 30 * 60 * 1000 # experiment duration in ms
vols_to_clip = 200
channels = ['green']
behaviors = ['dRotLabX', 'dRotLabY', 'dRotLabZ', 'speed']

#######################
### Loop over flies ###
#######################

for fly_idx, folder in enumerate(folders):
    
    ### Send email and define folder path ###
    function_durations = []
    print('Starting analysis of {}.'.format(folder))
    sys.stdout.flush()
    send_email('Starting {} ({} of {}).'.format(folder, fly_idx+1, len(folders)), 'wow')

    ### Load timestamps ###
    timestamps = load_timestamps(folder)
    timestamps = timestamps[vols_to_clip:,:]
    
    ### Load fictrac ###
    fictrac = load_fictrac(root_path, folder)
    
    ##########################
    ### Loop over channels ###
    ##########################

    for channel in channels:
        
        ### Load brain ###
        zbrain_file = os.path.join(folder, 'brain_zscored_' + channel + '.nii')
        try:
            # Try to load z-scored brain
            brain = load_numpy_brain(zbrain_file)
            dims = get_dims(brain)
        except:
            brain, dims = get_motcorr_brain(folder, channel=channel)

            # remove first bit of data since it often has some weirdness
            brain = brain[:,:,:,vols_to_clip:]
            dims['t'] = brain.shape[3]
            print('brain shape: {}'.format(brain.shape))

            ### Bleaching correction (per voxel) ###
            brain = bleaching_correction(brain)

            ### Z-score brain ###
            brain = z_score_brain(brain)
            save_brain(zbrain_file, brain)

        ###########################
        ### Loop over behaviors ###
        ###########################

        for behavior in behaviors:

            ### Prep given behavior ###
            fictrac_interp = prep_fictrac(fictrac, timestamps, fps, dur, behavior=behavior)
        
            ### Fit GLM ###
            scores, betas = fit_glm(brain, dims, fictrac_interp, beta_len)

            ### Save brain ###
            save_glm_map(scores, betas, folder, channel, behavior=behavior)

            print('Reached END.')
            sys.stdout.flush()
