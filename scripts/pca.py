import numpy as np
from time import time
import os
import sys
import scipy
from sklearn.decomposition import PCA

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

desired_flies = [21] # 1 index
fly_folders = [fly_folders[i-1] for i in desired_flies]
flies = [flies[i-1] for i in desired_flies]
print(fly_folders)
sys.stdout.flush()

folder = root_path + fly_folders[0]

z_brain_file = os.path.join(folder, 'brain_z.nii')

brain = load_numpy_brain(z_brain_file, channel=None, flip=False):

#every row is a time point of the whole brain (so flatten brain for each row)
X = np.reshape(brain, (-1, brain.shape[-1]))
X = np.swapaxes(X, 0, 1)

t0 = time()
pca = PCA()
PCs = pca.fit_transform(X)
print(time()-t0)

save_file = os.path.join(folder, 'pca_out')
np.save(save_file,PCs)