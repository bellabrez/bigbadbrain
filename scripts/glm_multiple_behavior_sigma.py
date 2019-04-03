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
from BigBadBrain.motcorr import get_motcorr_brain

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [27] # 1 index
deprecated_motcorr = False
folders = get_fly_folders(root_path, desired_flies)

beta_len = 21 #MUST BE ODD
fps = 50 #of fictrac camera
dur = 30 * 60 * 1000 # experiment duration in ms
vols_to_clip = 200
channels = ['green']
behaviors = ['speed'] #'dRotLabX', 'dRotLabY', 'speed'
fictrac_sigmas = [3]

#######################
### Loop over flies ###
#######################

for fly_idx, folder in enumerate(folders):
    
    ### Send email and define folder path ###
    function_durations = []
    print('\n~~~~ Starting analysis of {} ~~~~'.format(folder))
    sys.stdout.flush()
    send_email('Starting {} ({} of {}).'.format(folder, fly_idx+1, len(folders)), 'wow')

    ### Load timestamps ###
    timestamps = load_timestamps(folder)
    #timestamps = timestamps[vols_to_clip:,:]
    
    ### Load fictrac ###
    fictrac = load_fictrac(root_path, folder)
    
    ##########################
    ### Loop over channels ###
    ##########################

    for channel in channels:

        ##################
        ### Load brain ###
        ##################
        print('\n~~ Loading Brain ~~')
        sys.stdout.flush()
        zbrain_file = os.path.join(folder, 'brain_zscored_' + channel + '.nii')
        try:
            print('Trying to load z-scored brain.')
            # Try to load z-scored brain
            brain = load_numpy_brain(zbrain_file)
            dims = get_dims(brain)
        except:
            print('Failed. Trying to load motion corrected brain.')
            if deprecated_motcorr:
                deprecated_file = os.path.join(folder, 'motcorr.nii')
                brain = load_numpy_brain(deprecated_file)
                dims = get_dims(brain)
            else:
                brain, dims = get_motcorr_brain(folder, channel=channel)

            ### Bleaching correction (per voxel) ###
            brain = bleaching_correction(brain)

            ### Z-score brain ###
            brain = z_score_brain(brain)
            save_brain(zbrain_file, brain)

        ###########################
        ### Loop over behaviors ###
        ###########################

        for behavior in behaviors:
            for sigma in fictrac_sigmas:

                ### Prep given behavior ###
                fictrac_interp = prep_fictrac(fictrac, timestamps, fps, dur, behavior=behavior, sigma=sigma)
            
                ### Fit GLM ###
                scores, betas = fit_glm(brain, dims, fictrac_interp, beta_len)

                ### Save brain ###
                save_glm_map(scores, betas, folder, channel, behavior=behavior, fictrac_sigma=sigma)
