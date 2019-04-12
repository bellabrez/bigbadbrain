import numpy as np
import os
import sys
from time import time
import matplotlib.pyplot as plt

from BigBadBrain.brain import get_resolution, get_dims, load_numpy_brain, make_meanbrain
from BigBadBrain.utils import timing

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

def align_volume(fixed, moving, vol):
    """ Aligns a single 3D volume to another using antspy.

    Parameters
    ----------
    fixed: 3D antspy object to be warped to.
    moving: 4D antspy object to warp
    vol: int, time point of moving brain to warp

    Returns
    -------
    motCorr_vol: dictionary containing warped 3D antspy object

    """
    moving_vol = ants.from_numpy(moving[:,:,:,vol])
    motCorr_vol = ants.registration(fixed, moving_vol, type_of_transform='SyN')
    return motCorr_vol



@timing
def motion_correction(brain_master, brain_slave, directory, motcorr_directory):
    """ Performs non-linear warping of each red brain volume to the red temporal meanbrain,
    and duplicates this to the green channel.

    Parameters
    ----------
    brain_master: [y,x,z,t] numpy array of red channel
    brain_slave: [y,x,z,t] numpy array of green channel. This we be warped based on red channel.
    directory: full path of main fly folder
    motcorr_directory: full path to motcorr subdirectory

    Returns
    -------
    Nothing. """
    
    # Make mean brain
    meanbrain = ants.from_numpy(make_meanbrain(brain_master))
    dims = get_dims(brain_master)

    # Align each time volume to the meanbrain
    motCorr_brain_master = []
    motCorr_brain_slave = []
    transforms = []
    print('Performing motion correction...')
    sys.stdout.flush()
    for i in range(dims['t']):
        print('Aligning brain volume {} of {}...'.format(i+1, dims['t']), end='')
        sys.stdout.flush()
        t0 = time()
        
        #First, align given master volume to master meanbrain
        motCorr_vol_master = align_volume(fixed=meanbrain, moving=brain_master, vol=i)
        motCorr_brain_master.append(motCorr_vol_master['warpedmovout'].numpy())
        transforms.append(motCorr_vol_master['fwdtransforms'])
        
        #Then, use warp parameters on slave volume
        fixed = meanbrain
        moving = ants.from_numpy(brain_slave[:,:,:,i])
        transformlist = motCorr_vol_master['fwdtransforms']
        motCorr_brain_slave.append(ants.apply_transforms(fixed,moving,transformlist).numpy())
        
        print('Done. Duration: {:.1f}s'.format(time()-t0))
        sys.stdout.flush()

    # Save motcorr brains
    save_motCorr_brain(motCorr_brain_master, motcorr_directory, suffix='red')
    save_motCorr_brain(motCorr_brain_slave, motcorr_directory, suffix='green')

    transform_matrix = save_transform_files(transforms, motcorr_directory)
    save_motion_figure(transform_matrix, director, motcorr_directory)

@timing
def save_transform_files(transforms, motcorr_directory)
# Organize mat transform file
    transform_matrix = []
    for i, transform in enumerate(transforms):
        for x in transform:
            if '.mat' in x:
                temp = ants.read_transform(x)
                transform_matrix.append(temp.parameters)
    transform_matrix = np.array(transform_matrix)

    # Save mat transform file
    save_file = os.path.join(motcorr_directory, 'motcorr_params')
    np.save(save_file,transform_matrix)
    return transform_matrix

@timing
def save_motion_figure(transform_matrix, directory, motcorr_directory):
    # Get voxel resolution for figure
    file = os.path.join(directory, 'functional.xml')
    x_res, y_res, z_res = get_resolution(file)

    # Save figure of motion over time
    save_file = os.path.join(motcorr_directory, 'motion_correction.png')
    plt.figure(figsize=(10,10))
    plt.plot(transform_matrix[:,9]*x_res, label = 'y') # note, resolutions are switched since axes are switched
    plt.plot(transform_matrix[:,10]*y_res, label = 'x')
    plt.plot(transform_matrix[:,11]*z_res, label = 'z')
    plt.ylabel('Motion Correction, um')
    plt.xlabel('Time')
    plt.title(directory)
    plt.legend()
    plt.savefig(save_file, bbox_inches='tight', dpi=300)

@timing
def save_motCorr_brain(brain, directory, suffix):
    """ Saves a 4D motion corrected brain.

    Parameters
    ----------
    brain: 4D antspy brain
    directory: full directory in which to save the brain
    suffix: str to add to file name

    Returns
    -------
    motCorr_brain_ants: 4D motion corrected ants brain

    """
    brain = np.moveaxis(np.asarray(brain),0,3)
    motCorr_brain_ants = ants.from_numpy(brain)
    save_file = os.path.join(directory, 'motcorr_' + suffix + '.nii')
    ants.image_write(motCorr_brain_ants, save_file)
    return motCorr_brain_ants

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