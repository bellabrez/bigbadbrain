import os
import sys
import numpy as np

from .utils import timing
from .brain import load_numpy_brain, bleaching_correction, z_score_brain, save_brain
from .glm import motion_correction

@timing
def get_motcorr_brain(directory, channel):
    """ Tries to load motcorr_brain, otherwise calls motion_correction to create it.
    Parameters
    ----------
    directory: full path
    channel: str, 'red' (channel 0) or 'green' (channel 1)

    Returns
    -------
    brain: [y,x,z,t] numpy array containing motion corrected brain

    """
    ### Make subfolder if it doesn't exist
    subfolder = 'motcorr'
    motcorr_directory = os.path.join(directory, subfolder)
    if not os.path.exists(motcorr_directory):
        os.makedirs(motcorr_directory)

    # try to open motion corrected brain, else make it
    try:
        print('Trying to load {} motion-corrected brain.'.format(channel))
        sys.stdout.flush()

        brain_file = os.path.join(motcorr_directory, 'motcorr_{}.nii'.format(channel))
        brain = load_numpy_brain(brain_file)

        print('Loaded {} motion-corrected brain.'.format(channel))
        sys.stdout.flush()

        return brain

    except:
        print('Failed to load {} motion-corrected brain.'.format(channel))
        print('Trying to load functional brain.')
        sys.stdout.flush()

        brain_file = os.path.join(directory, 'functional.nii')
        brain = load_numpy_brain(brain_file)

        print('Loaded functional brain.')
        print('Brain shape: {}'.format(np.shape(brain)))
        sys.stdout.flush()

        ### Perform motion correction ###
        brain = motion_correction(brain_master=brain[:,:,:,:,0],
                                  brain_slave=brain[:,:,:,:,1],
                                  directory=directory,
                                  motcorr_directory=motcorr_directory)
        
        brain_file = os.path.join(motcorr_directory, 'motcorr_{}.nii'.format(channel))
        brain = load_numpy_brain(brain_file)

        return brain

@timing
def get_z_brain(directory, channel):
    zbrain_file = os.path.join(directory, 'brain_zscored_' + channel + '.nii')
    try:
        print('Trying to load z-scored brain.')
        brain = load_numpy_brain(zbrain_file)
    except:
        print('Failed. Trying to load motion corrected brain.')
        brain = get_motcorr_brain(directory, channel=channel)

        ### Bleaching correction (per voxel) ###
        brain = bleaching_correction(brain)

        ### Z-score brain ###
        brain = z_score_brain(brain)
        save_brain(zbrain_file, brain)
    return brain