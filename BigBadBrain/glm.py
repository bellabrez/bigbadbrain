import sys
import os
import numpy as np
from scipy.linalg import toeplitz
from sklearn.linear_model import LassoLarsIC

from BigBadBrain.utils import timing

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

@timing
def fit_visual_glm(brain, dims, stimulus, timestamps, bins):
    print('\n~~ Fitting GLM ~~')
    print('Z-slice progress (out of {}): '.format(dims['z']), end='')
    sys.stdout.flush()

    stimuli_times = stimulus['times']
    betas = []
    scores = []

    for z in range(dims['z']):

            ### Printing updates ###
            if z == dims['z']-1:
                print('{}.'.format(z))
                sys.stdout.flush()
            else:
                print('{}, '.format(z), end = '')
                sys.stdout.flush()

            voxel_times = timestamps[:,z]
            X = create_visual_X(stimuli_times, voxel_times, bins)
            
            for x in range(dims['x']):
                for y in range(dims['y']):
                    Y = brain[y,x,z,:]
                    model = LassoLarsIC(criterion='bic')
                    model.fit(X, Y)
                    betas.append(model.coef_)
                    scores.append(model.score(X,Y))
    print('shape of betas: {}'.format(np.shape(betas)))
    print('len of bins: {}'.format(len(bins)))
    scores = np.reshape(scores, (dims['z'], dims['x'], dims['y']))
    betas = np.reshape(betas, (dims['z'], dims['x'], dims['y'], len(bins)-1))
    return scores, betas

def create_bins(bin_size,pre_dur,post_dur):
    bins_pre = np.flip(np.arange(0,pre_dur-1,-bin_size),axis=0)
    bins_post = np.arange(0,post_dur+1,bin_size)
    bins = np.unique(np.concatenate((bins_pre, bins_post)))
    return bins

def create_visual_X(stimuli_times, voxel_times, bins):
    ### Get vector that assigns voxel activities to bins ###
    # Create real-time bins for each stimulus presentation
    time_bins = np.add.outer(stimuli_times,bins)
    time_bins_flat = np.reshape(time_bins,time_bins.shape[0]*time_bins.shape[1])
    # Find which bin each voxel timestamp belongs to
    binned = np.searchsorted(time_bins_flat, voxel_times)
    # The mod will give numbers that match bin numbers
    bin_mod = binned%21

    ### Use bin vector to create X matrix ###
    X = np.zeros((len(bin_mod), len(bins)))
    # This is used to correct index into X
    axis = np.arange(0,len(bin_mod))
    # For each row of X, put a 1 in the correct column (bin) if that row gets one
    X[axis,bin_mod] = 1
    # Remove first column, which is nonsense
    X = X[:,1:]
    return X

@timing
def fit_glm(brain, dims, fictrac, beta_len, single_slice=False):
    print('\n~~ Fitting GLM ~~')
    print('Z-slice progress (out of {}): '.format(dims['z']), end='')
    sys.stdout.flush()
    sys.stdout.flush()
    middle = int((beta_len - 1) / 2)
    betas = []
    scores = []
    for z in range(dims['z']):

        ### Printing updates ###
        if z == dims['z']-1:
            print('{}.'.format(z))
            sys.stdout.flush()
        else:
            print('{}, '.format(z), end = '')
            sys.stdout.flush()

        Y = fictrac[:,z]
        for x in range(dims['x']):
            for y in range(dims['y']):
                if single_slice:
                    voxel_activity = brain[y,x,:]
                else:
                    voxel_activity = brain[y,x,z,:]
                X = toeplitz(voxel_activity, np.zeros(beta_len))
                X = np.roll(X, middle)
                model = LassoLarsIC(criterion='bic')
                model.fit(X, Y)
                betas.append(model.coef_)
                scores.append(model.score(X,Y))
    scores = np.reshape(scores, (dims['z'], dims['x'], dims['y']))
    betas = np.reshape(betas, (dims['z'], dims['x'], dims['y'], beta_len))
    return scores, betas

@timing
def save_glm_map(scores_vol, betas_vol, folder, channel, param=None):
    print('\n~~ Saving GLM ~~')
    sys.stdout.flush()

    # Make subfolder if it doesn't exist
    subfolder = 'glm'
    directory = os.path.join(folder, subfolder)
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    # Save scores
    file = 'multivariate_analysis_' + channel + '_' + param + '.nii'
    save_file = os.path.join(directory, file)
    brain_to_save = np.swapaxes(scores_vol, 0, 2)
    ants.image_write(ants.from_numpy(brain_to_save), save_file)

    # Save betas
    file = 'multivariate_analysis_betas_' + channel + '_' + param + '.nii'
    save_file = os.path.join(directory, file)
    brain_to_save = np.swapaxes(betas_vol, 0, 2)
    ants.image_write(ants.from_numpy(brain_to_save), save_file)

@timing
def create_multivoxel_X_matrix(brain, dims, beta_len):
    print('\n~~ Creating Multivoxel X matrix ~~')
    sys.stdout.flush()
    middle = int((beta_len - 1) / 2)
    Xs = []
    print('Z-slice progress (out of {}): '.format(dims['z']), end='')
    sys.stdout.flush()
    for z in range(dims['z']):

        ### Printing updates ###
        if z == dims['z']-1:
            print('{}.'.format(z))
            sys.stdout.flush()
        else:
            print('{}, '.format(z), end = '')
            sys.stdout.flush()

        for x in range(dims['x']):
            for y in range(dims['y']):
                voxel_activity = brain[y,x,z,:]
                X = toeplitz(voxel_activity, np.zeros(beta_len))
                X = np.roll(X, middle)
                Xs.append(X)
    out = np.concatenate(Xs,axis=1)
    return out