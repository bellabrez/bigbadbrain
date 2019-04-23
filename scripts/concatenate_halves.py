import numpy as np
from time import time
import os
import sys
import scipy
import psutil
sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants
import BigBadBrain as bbb

first_half = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_30/optic/motcorr/motcorr_green_first_half.nii'
second_half = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_30/optic/motcorr/motcorr_green_second_half.nii'
brain1 = bbb.load_numpy_brain(first_half)
brain2 = bbb.load_numpy_brain(second_half)
brain = np.concatenate((brain1,brain2),axis=-1)
brain1 = None
brain2 = None
save_file = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/fly_30/optic/motcorr/motcorr_green.nii'
bbb.save_brain(save_file, brain)
memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
print('Current memory usage: {:.2f}GB'.format(memory_usage))
sys.stdout.flush()