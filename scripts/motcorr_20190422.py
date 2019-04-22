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
root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
desired_flies = [30] # 1 index
folders = bbb.get_fly_folders(root_path, desired_flies)

######################################
### What brain areas and channels? ###
######################################
channels = ['green', 'red']
brain_regions = ['optic'] # if not nested, put ''

#############
### START ###
#############
print('motcorr only')
sys.stdout.flush()
for fly_idx, folder in enumerate(folders):
    for brain_region in brain_regions:

        directory = os.path.join(folder, brain_region)
        bbb.announce_start(directory, fly_idx, folders)
        
        for channel in channels:
            brain = bbb.get_z_brain(directory, channel)