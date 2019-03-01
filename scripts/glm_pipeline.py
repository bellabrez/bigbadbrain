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
from bigbrain.fictrac import load_fictrac
from bigbrain.fictrac import prep_fictrac
from bigbrain.utils import load_timestamps
from bigbrain.utils import sort_nicely
from bigbrain.utils import send_email
from bigbrain.glm import fit_glm
from bigbrain.glm import save_glm_map
from bigbrain.motcorr import get_motcorr_brain

class Fly:
    def __init__(self):
        pass
    
root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'

fly_folders = sorted(os.listdir(root_path))
fly_folders = [x for x in fly_folders if 'fly' in x]
sort_nicely(fly_folders)

flies = [Fly() for i in range(len(fly_folders))]

print('Created flies from folders {}'.format(fly_folders))
sys.stdout.flush()

desired_flies = [20] # 1 index
fly_folders = [fly_folders[i-1] for i in desired_flies]
flies = [flies[i-1] for i in desired_flies]
print(fly_folders)
sys.stdout.flush()

beta_len = 21 #MUST BE ODD
fps = 50 #of fictrac camera
dur = 30 * 60 * 1000 # experiment duration in ms
vols_to_clip = 200

for fly_idx, fly in enumerate(flies):
    
    ### Send email and define folder path ###
    function_durations = []
    print('Starting analysis of {}.'.format(fly_folders[fly_idx]))
    sys.stdout.flush()
    send_email('Starting {} ({} of {}).'.format(fly_folders[fly_idx], fly_idx+1, len(flies)), 'wow')
    folder = root_path + fly_folders[fly_idx]

    ### Load timestamps ###
    timestamps = load_timestamps(folder)
    
    ### Load fictrac (and prep) ###
    # add sanity check for failed frames. Simply check if and speed measurements are over 10
    fictrac = load_fictrac(root_path, fly_folders[fly_idx])
    print('About to prep')
    sys.stdout.flush()
    fictrac_interp = prep_fictrac(fictrac, timestamps, fps, dur)
    print('Just preped')
    sys.stdout.flush()
    
    # remove first bit of data since it often has some weirdness
    timestamps = timestamps[vols_to_clip:,:]
    fictrac_interp = fictrac_interp[vols_to_clip:,:]
    
    send_email('loaded timestamps and fictrac', 'wow')
    
    ############# Do remaining analysis on both brain channels ###############
    channels = ['green', 'red']
    for channel in channels:
        ### Load brain ###
        brain, dims = get_motcorr_brain(folder, channel=channel)

        # remove first bit of data since it often has some weirdness
        brain = brain[:,:,:,vols_to_clip:]
        dims['t'] = brain.shape[3]

        ### Bleaching correction (per voxel) ###
        brain = bleaching_correction(brain)

        ### Z-score brain ###
        brain = z_score_brain(brain)

        ### Fit GLM ###
        scores = fit_glm(brain, dims, fictrac_interp, beta_len)

        ### Save brain ###
        save_glm_map(scores, folder, channel)

        ### Send email ###

        # Prep timing string
        func_str = ''
        for func in function_durations:
            func_str += '{} ===== {:.2f} min\n'.format(func['name'], func['duration'])
        print(func_str)
        sys.stdout.flush()

        #send_email('Success {} ({} of {}):{} {} channel.'.format(fly_folders[fly_idx], fly_idx+1, len(flies)), func_str, channel)
        print('Reached END.')
        sys.stdout.flush()
