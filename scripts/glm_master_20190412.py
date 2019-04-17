import numpy as np
from time import time
import os
import sys
import scipy

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

from BigBadBrain.brain import bleaching_correction, z_score_brain, get_resolution, save_brain, load_numpy_brain, get_dims
from BigBadBrain.fictrac import load_fictrac, interpolate_fictrac
from BigBadBrain.utils import load_timestamps, get_fly_folders, send_email, announce_start, timing
from BigBadBrain.glm import fit_glm, save_glm_map
from BigBadBrain.motcorr import get_motcorr_brain

@timing
def get_z_brain(directory, channel):
    zbrain_file = os.path.join(directory, 'brain_zscored_' + channel + '.nii')
    try:
        print('Trying to load z-scored brain.')
        brain = load_numpy_brain(zbrain_file)
    except:
        print('Failed. Trying to load motion corrected brain.')
        brain = get_motcorr_brain(directory, channel=channel)

        ### Bleaching correction (per voxel) ###
        brain = bleaching_correction(brain)

        ### Z-score brain ###
        brain = z_score_brain(brain)
        save_brain(zbrain_file, brain)
    return brain

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [29, 30] # 1 index
folders = get_fly_folders(root_path, desired_flies)

beta_len = 21 #MUST BE ODD
fps = 50 #of fictrac camera
dur = 30 * 60 * 1000 # experiment duration in ms
vols_to_clip = 200
channels = ['green']
behaviors = ['speed'] #'dRotLabX', 'dRotLabY', 'speed'
fictrac_sigmas = [3]
brain_regions = ['central'] # if None, make it look for folders and do all of them

for fly_idx, folder in enumerate(folders):
    for brain_region in brain_regions:

        directory = os.path.join(folder, brain_region)
        announce_start(directory, fly_idx, folders)
        timestamps = load_timestamps(directory)
        fictrac = load_fictrac(directory)
        
        for channel in channels:
            brain = get_z_brain(directory, channel)
            dims = get_dims(brain)
            for behavior in behaviors:
                for sigma in fictrac_sigmas:

                    ### Prep given behavior ###
                    fictrac_interp = interpolate_fictrac(fictrac, timestamps, fps, dur, behavior=behavior, sigma=sigma)
                
                    ### Fit GLM ###
                    scores, betas = fit_glm(brain, fictrac_interp, beta_len)

                    ### Save brain ###
                    save_glm_map(scores, betas, directory, channel, param=behavior)
