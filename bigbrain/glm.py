import sys
import os
import numpy as np
from scipy.linalg import toeplitz
from sklearn.linear_model import LassoLarsIC

from bigbrain.utils import timing

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

@timing
def fit_glm(brain, dims, fictrac, beta_len):
    print('Fit GLM.')
    sys.stdout.flush()
    middle = int((beta_len - 1) / 2)
    betas = []
    scores = []
    for z in range(dims['z']):
        print('~~ z:{} ~~ '.format(z), end = '')
        sys.stdout.flush()
        Y = fictrac[:,z]
        for x in range(dims['x']):
            for y in range(dims['y']):
                voxel_activity = brain[y,x,z,:]
                X = toeplitz(voxel_activity, np.zeros(beta_len))
                X = np.roll(X, middle)
                model = LassoLarsIC(criterion='bic')
                model.fit(X, Y)
                betas.append(model._coef)
                scores.append(model.score(X,Y))
    scores = np.reshape(scores, (dims['z'], dims['x'], dims['y']))
    betas = np.reshape(betas, (dims['z'], dims['x'], dims['y'], -1))
    return scores, betas

def save_glm_map(scores_vol, betas_vol, folder, channel, behavior='speed'):
    print('Saving glm vol.')
    sys.stdout.flush()

    # Make subfolder if it doesn't exist
    subfolder = 'glm'
    directory = os.path.join(folder, subfolder)
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    # Save scores
    file = 'multivariate_analysis_' + channel + '_' + behavior + '.nii'
    save_file = os.path.join(directory, file)
    brain_to_save = np.swapaxes(scores_vol, 0, 2)
    ants.image_write(ants.from_numpy(brain_to_save), save_file)

    # Save betas
    file = 'multivariate_analysis_betas_' + channel + '_' + behavior + '.nii'
    save_file = os.path.join(directory, file)
    brain_to_save = np.swapaxes(betas_vol, 0, 2)
    ants.image_write(ants.from_numpy(brain_to_save), save_file)
