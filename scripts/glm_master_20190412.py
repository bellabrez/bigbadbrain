import numpy as np
from time import time
import os
import sys
import scipy
sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants
import BigBadBrain as bbb

##########################
### What flies to run? ###
##########################
sys.stdout.flush()
root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [30] # 1 index
folders = bbb.get_fly_folders(root_path, desired_flies)

######################################
### What brain areas and channels? ###
######################################
channels = ['green']
brain_regions = ['optic'] # if not nested, put ''

################################
### Perform visual analysis? ###
################################
visual = True
bin_size = 100 #in ms
pre_dur = 500 #in ms
post_dur = 1500 #in ms

####################################
### Perform behavioral analysis? ###
####################################
behavior = False
use_abs_value = True # Takes the abs value of the behavior
behaviors = ['dRotLabY','dRotLabZ', 'dRotLabX', 'my_speed', 'speed_all_3'] #'dRotLabX', 'dRotLabY', 'speed'
fictrac_sigmas = [3]
beta_len = 21 #MUST BE ODD
fps = 50 #of fictrac camera
dur = 30 * 60 * 1000 # experiment duration in ms

#############
### START ###
#############
for fly_idx, folder in enumerate(folders):
    for brain_region in brain_regions:

        directory = os.path.join(folder, brain_region)
        bbb.announce_start(directory, fly_idx, folders)
        timestamps = bbb.load_timestamps(directory)

        if behavior:
            fictrac = bbb.load_fictrac(directory)
        if visual:
            unique_stimuli = bbb.get_stimuli(directory)
        
        for channel in channels:
            brain = bbb.get_z_brain(directory, channel)
            dims = bbb.get_dims(brain)

            if behavior:
                for behavior in behaviors:
                    for sigma in fictrac_sigmas:

                        ### Prep given behavior ###
                        fictrac_interp = bbb.interpolate_fictrac(fictrac,
                                                                 timestamps,
                                                                 fps,
                                                                 dur,
                                                                 behavior=behavior,
                                                                 sigma=sigma,
                                                                 use_abs_value=use_abs_value)

                        ### Fit GLM ###
                        scores, betas = bbb.fit_glm(brain, fictrac_interp, beta_len)

                        ### Save brain ###
                        bbb.save_glm_map(scores, betas, directory, channel, param=behavior+'abs')

            if visual:
                for stimulus in unique_stimuli:

                    ### Fit GLM ###
                    scores, betas = bbb.fit_visual_glm(brain, stimulus, timestamps, bin_size, pre_dur, post_dur)

                    ### Save brain ###
                    bbb.save_glm_map(scores, betas, directory, channel, param=str(stimulus['angle']))

            if behavior and visual:
                for stimulus in unique_stimuli:
                
                    ### Create behavior STA plot ###
                    bbb.create_stim_triggered_behavior_plot(fictrac, stimulus, directory)
