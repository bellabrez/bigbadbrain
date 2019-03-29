import numpy as np
from time import time
import os
import sys
import scipy
from sklearn.decomposition import PCA

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

from BigBadBrain.brain import bleaching_correction, z_score_brain, get_resolution, load_numpy_brain
from BigBadBrain.utils import load_timestamps, sort_nicely, send_email

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

desired_flies = [25] # 1 index
fly_folders = [fly_folders[i-1] for i in desired_flies]
flies = [flies[i-1] for i in desired_flies]
print(fly_folders)
sys.stdout.flush()

folder = root_path + fly_folders[0]

### Load Brain ###

z_brain_file = os.path.join(folder, 'brain_zscored_green.nii')
try:
    brain = load_numpy_brain(z_brain_file)
except:
    brain_file = os.path.join(folder, 'motcorr', 'brain_green.nii')
    brain = load_numpy_brain(brain_file)
    brain = bleaching_correction(brain)
    brain = z_score_brain(brain)
    save_brain(zbrain_file, brain)

### Perform PCA ###

for swap in [True, False]:

    #every row is a time point of the whole brain (so flatten brain for each row)
    X = np.reshape(brain, (-1, brain.shape[-1]))

    if swap:
        X = np.swapaxes(X, 0, 1)

    t0 = time()
    pca = PCA(n_components=2)
    #PCs = pca.fit_transform(X)
    pca.fit(X)

    print(time()-t0)

    ### Save Output ###

    # Make subfolder if it doesn't exist
    subfolder = 'pca'
    directory = os.path.join(folder, subfolder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_file = os.path.join(directory, 'pca_explained_variance_ratio' + str(swap))
    np.save(save_file,np.asarray(pca.explained_variance_ratio_))

    save_file = os.path.join(directory, 'pca_components' + str(swap))
    np.save(save_file,np.asarray(pca.components_))

    save_file = os.path.join(directory, 'pca_singular_values' + str(swap))
    np.save(save_file,np.asarray(pca.singular_values_))