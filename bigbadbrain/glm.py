import sys
import os
import numpy as np
from scipy.linalg import toeplitz
from sklearn.linear_model import LassoLarsIC

from bigbadbrain.utils import timing, create_bins
from bigbadbrain.brain import get_dims

import platform
if platform.system() != 'Windows':
    sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
    import ants

@timing
def fit_visual_glm(brain, stimulus, timestamps, bin_size, pre_dur, post_dur):
    """ Will fit a cross-validated Generalized Linear Model of the relationship between
    visual stimuli and neural activity.

    More detail to be added later.

    Parameters
    ----------
    brain: numpy array [y,x,z,t].
    stimulus: dictionary of stimulus parameters. Must contain a list of stimulus times called 'times'.
    timestamps: numpy array of times of Bruker frames. [t,z].
    bin_size: resolution over which to bin voxel responses (in ms).
    pre_dur: time before stimuli start to begin glm (in ms).
    post_dur: time after stimuli start to end glm (in ms).

    Returns
    -------
    scores: [y,x,z] numpy array of R2 scores for each voxel.
    betas: [y,x,z,b] numpy array of betas for each voxel. Betas calculated based on bin parameters.

    """

    bins = create_bins(bin_size,pre_dur,post_dur)
    dims = get_dims(brain)
    stimuli_times = stimulus['times']
    betas = []
    scores = []

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
    scores = np.swapaxes(scores, 0, 2)
    betas = np.reshape(betas, (dims['z'], dims['x'], dims['y'], len(bins)-1))
    betas = np.swapaxes(betas, 0, 2)
    return scores, betas

def create_visual_X(stimuli_times, voxel_times, bins):
    """ Helper function to create X matrix for visual stimuli GLM.

    Parameters
    ----------
    stimuil_times: list of times (in ms) of stimuli onsets.
    voxel_times: 1D numpy array of times (in ms) of voxel collection times.
    bins: 1D numpy array of desired bins (see create_bins).

    Returns
    -------
    X: [t,bins] numpy array. Used in fit_visual_glm. All zeros except one where a given voxel collection time
    (row) falls within a given bin (columns). """

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
def fit_glm(brain, fictrac, beta_len, single_slice=False):
    """ Will fit a cross-validated Generalized Linear Model of the relationship between 
    behavior and neural activity.

    Will add more detail later.

    Parameters
    ----------
    brain: numpy array [y,x,z,t].
    fictrac: [t,z] numpy array of values of a given behavior at times interpolated from timestamps (t,z).
    beta_len: odd int of number of indicies to calculate betas for. Times of indicies are based on imaging rate.
    Will equally sample both sides of zero.
    single_slice: Defaults to False. If True, this function can be run on a brain of shape [y,x,t].

    Returns
    -------
    scores: [y,x,z] numpy array of R2 scores for each voxel.
    betas: [y,x,z,b] numpy array of betas for each voxel. Betas calculated based on bin parameters.

    """
    
    dims = get_dims(brain)
    print('Z-slice progress (out of {}): '.format(dims['z']), end='')
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
    scores = np.swapaxes(scores, 0, 2)
    betas = np.reshape(betas, (dims['z'], dims['x'], dims['y'], beta_len))
    betas = np.swapaxes(betas, 0, 2)
    return scores, betas

@timing
def save_glm_map(scores, betas, directory, channel, param=None):
    """ Will save scores and betas (outputs of glms) into nifti files.

    Creates and saves in subdirectory "glm".

    Parameters
    ----------
    scores: [y,x,z] numpy array of R2 scores for each voxel.
    betas: [y,x,z,b] numpy array of betas for each voxel.
    directory: string of full path to directory into which to save.
    channel: string to include in file save name.
    param: string to include in save name.

    Returns
    -------
    Nothing. """

    # Make subfolder if it doesn't exist
    subfolder = 'glm'
    directory = os.path.join(directory, subfolder)
    if not os.path.exists(directory):
        os.makedirs(directory)
 
    # Save scores
    file = 'multivariate_analysis_' + channel + '_' + param + '.nii'
    save_file = os.path.join(directory, file)
    ants.image_write(ants.from_numpy(scores), save_file)

    # Save betas
    file = 'multivariate_analysis_betas_' + channel + '_' + param + '.nii'
    save_file = os.path.join(directory, file)
    ants.image_write(ants.from_numpy(betas), save_file)

@timing
def create_multivoxel_X_matrix(brain, dims, beta_len):
    """ In progress attemping to do multivoxel predictions.

    Parameters
    ----------

    Returns
    -------
    """

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

@timing
def create_multivoxel_single_X_matrix(brain):
    """ In progress attemping to do multivoxel predictions.

    Parameters
    ----------

    Returns
    -------
    """
    X = np.reshape(brain, (-1,brain.shape[-1]))
    return X

@timing
def fit_all_voxel_glm(brain, fictrac):
    
    dims = get_dims(brain)
    
    X = np.swapaxes(np.reshape(brain,(-1,brain.shape[-1])),0,1)
    print('X shape: {}'.format(X.shape))

    z = 0
    Y = fictrac[:,z]
    model = LassoLarsIC(criterion='bic')
    model.fit(X, Y)

    score = model.score(X,Y)

    print('betas shape: {}'.format(np.shape(model.coef_)))
    sys.stdout.flush()

    betas = np.reshape(model.coef_, (dims['y'], dims['x'], dims['z']))

    print('betas shape: {}'.format(np.shape(betas)))
    sys.stdout.flush()

    return score, betas