import numpy as np
from time import time
import os
import sys
import scipy
sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants
import BigBadBrain as bbb

@timing
def get_z_brain(directory, channel):
    zbrain_file = os.path.join(directory, 'brain_zscored_' + channel + '.nii')
    try:
        print('Trying to load z-scored brain.')
        brain = bbb.load_numpy_brain(zbrain_file)
    except:
        print('Failed. Trying to load motion corrected brain.')
        brain = bbb.get_motcorr_brain(directory, channel=channel)

        ### Bleaching correction (per voxel) ###
        brain = bbb.bleaching_correction(brain)

        ### Z-score brain ###
        brain = bbb.z_score_brain(brain)
        bbb.save_brain(zbrain_file, brain)
    return brain

root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [29, 30] # 1 index
folders = bbb.get_fly_folders(root_path, desired_flies)

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
        bbb.announce_start(directory, fly_idx, folders)
        timestamps = bbb.load_timestamps(directory)
        fictrac = bbb.load_fictrac(directory)
        
        for channel in channels:
            brain = get_z_brain(directory, channel)
            dims = bbb.get_dims(brain)
            for behavior in behaviors:
                for sigma in fictrac_sigmas:

                    ### Prep given behavior ###
                    fictrac_interp = bbb.interpolate_fictrac(fictrac, timestamps, fps, dur, behavior=behavior, sigma=sigma)
                
                    ### Fit GLM ###
                    scores, betas = bbb.fit_glm(brain, fictrac_interp, beta_len)

                    ### Save brain ###
                    bbb.save_glm_map(scores, betas, directory, channel, param=behavior)
