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
from BigBadBrain.glm import fit_visual_glm, save_glm_map, create_bins
from BigBadBrain.motcorr import get_motcorr_brain
from BigBadBrain.visual import load_photodiode, load_visual_stimuli_data, parse_stim_starts_photodiode, set_unique_stimuli_display_times

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [28] # 1 index
deprecated_motcorr = False
folders = get_fly_folders(root_path, desired_flies)

channels = ['green']
bin_size = 100 #in ms
pre_dur = -500 #in ms
post_dur = 1500 #in ms
bins = create_bins(bin_size,pre_dur,post_dur)

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
    
    ### Load visual stimuli ###
    vision_path = os.path.join(folder,'visual')
    t,pd1,pd2 = load_photodiode(vision_path)
    stimuli, unique_stimuli = load_visual_stimuli_data(vision_path)
    stimuli_starts = parse_stim_starts_photodiode(pd1,stimuli)
    unique_stimuli = set_unique_stimuli_display_times(unique_stimuli, stimuli, stimuli_starts)
    
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
            print('Success.')
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
        ### Loop over stimuli ###
        ###########################

        for stimulus in unique_stimuli:

            ### Fit GLM ###
            scores, betas = fit_visual_glm(brain, dims, stimulus, timestamps, bins)

            ### Save brain ###
            save_glm_map(scores, betas, folder, channel, param=str(stimulus['angle']))
