import numpy as np
import os
import sys
from time import time
import matplotlib.pyplot as plt

from bigbrain.brain import get_resolution
from bigbrain.brain import get_dims
from bigbrain.brain import load_numpy_brain
from bigbrain.utils import timing

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

def align_volume(fixed, moving, vol):        
    moving_vol = ants.from_numpy(moving[:,:,:,vol])
    motCorr_vol = ants.registration(fixed, moving_vol, type_of_transform='SyN')
    return motCorr_vol

def save_motCorr_brain(brain, folder, suffix):
    brain = np.moveaxis(np.asarray(brain),0,3)
    motCorr_brain_ants = ants.from_numpy(brain)
    save_file = os.path.join(folder, 'motcorr_' + suffix + '.nii')
    ants.image_write(motCorr_brain_ants, save_file)
    return motCorr_brain_ants

@timing
def motion_correction(brain_master=None, brain_slave=None, folder=None, subfolder=None):
    directory = os.path.join(folder, subfolder)
    if brain_master is None:
        raise Exception('Must supply brain_master.')
    elif brain_master is not None and np.shape(brain_master) != np.shape(brain_slave):
        raise Exception('Dimensions of master and slave brain must match.')
    
    # Make mean brain
    print('Creating meanbrain...', end='')
    sys.stdout.flush()
    t = time()
    meanbrain = ants.from_numpy(np.mean(brain_master, axis=-1))
    print('Done. Duration: {:.1f}s'.format(time()-t))
    sys.stdout.flush()

    # Align each time volume to the meanbrain
    motCorr_brain_master = []
    motCorr_brain_slave = []
    transforms = []

    print('Performing motion correction...')
    sys.stdout.flush()
    for i in range(np.shape(brain_master)[3]):
        print('Aligning brain volume {} of {}...'.format(i+1, np.shape(brain_master)[3]), end='')
        sys.stdout.flush()
        t0 = time()
        
        #First, align given master volume to master meanbrain
        motCorr_vol_master = align_volume(fixed=meanbrain, moving=brain_master, vol=i)
        motCorr_brain_master.append(motCorr_vol_master['warpedmovout'].numpy())
        transforms.append(motCorr_vol_master['fwdtransforms'])
        
        #Then, use warp parameters on slave volume
        if brain_slave is not None:
            fixed = meanbrain
            moving = ants.from_numpy(brain_slave[:,:,:,i])
            transformlist = motCorr_vol_master['fwdtransforms']
            motCorr_brain_slave.append(ants.apply_transforms(fixed,moving,transformlist).numpy())
        
        print('Done. Duration: {:.1f}s'.format(time()-t0))
        sys.stdout.flush()

    # Save motcorr brain(s)
    print('Saving brain...', end='')
    sys.stdout.flush()
    t = time()
    brain_master_motCorr = save_motCorr_brain(motCorr_brain_master, directory, suffix='red')
    
    if brain_slave is not None:
        brain_slave_motCorr = save_motCorr_brain(motCorr_brain_slave, directory, suffix='green')
    print('Done. Duration: {:.1f}s'.format(time()-t))
    sys.stdout.flush()

    # Organize mat transform file
    print('Organizing transform file.')
    sys.stdout.flush()
    transform_matrix = []
    for i, transform in enumerate(transforms):
        for x in transform:
            if '.mat' in x:
                temp = ants.read_transform(x)
                transform_matrix.append(temp.parameters)
    transform_matrix = np.array(transform_matrix)

    # Save mat transform file
    print('Saving transform file.')
    sys.stdout.flush()
    save_file = os.path.join(directory, 'motcorr_params')
    np.save(save_file,transform_matrix)

    # Get voxel resolution for figure
    print('Getting voxel resolution.')
    sys.stdout.flush()
    file = os.path.join(folder, 'functional.xml')
    x_res, y_res, z_res = get_resolution(file)

    # Save figure of motion over time
    print('Saving motion correction figure.')
    sys.stdout.flush()
    save_file = os.path.join(directory, 'motion_correction.png')
    plt.figure(figsize=(10,10))
    plt.plot(transform_matrix[:,9]*x_res, label = 'y') # note, resolutions are switched since axes are switched
    plt.plot(transform_matrix[:,10]*y_res, label = 'x')
    plt.plot(transform_matrix[:,11]*z_res, label = 'z')
    plt.ylabel('Motion Correction, um')
    plt.xlabel('Time')
    plt.title(folder)
    plt.legend()
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    
    if brain_slave is not None:
        return brain_master_motCorr, brain_slave_motCorr
    else:
        return brain_master_motCorr

@timing
def get_motcorr_brain(folder, channel=None):
    # Make subfolder if it doesn't exist
    subfolder = 'motcorr'
    directory = os.path.join(folder, subfolder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # If it exists, load motion correction brain, else make it
    if channel == 'green':
        # try to open motcorr_green, else make it (and make red)
        try:
            print('Trying to load green motion-corrected brain.')
            sys.stdout.flush()
            brain_file = os.path.join(directory, 'motcorr_green.nii')
            brain_green = load_numpy_brain(brain_file)
            print('Loaded green motion-corrected brain.')
            sys.stdout.flush()
        except:
            print('Failed to load green motion-corrected brain.')
            print('Trying to load functional brain.')
            sys.stdout.flush()
            brain_file = folder + '/functional.nii'
            brain_green = load_numpy_brain(brain_file, channel='green')
            brain_red = load_numpy_brain(brain_file, channel='red')
            print('Loaded green and red functional brains.')
            sys.stdout.flush()

            ### Perform motion correction ###
            print('Performing motion-correction.')
            sys.stdout.flush()
            brain_red, brain_green = motion_correction(brain_master=brain_red, brain_slave=brain_green, folder=folder, subfolder=subfolder)
        
        dims = get_dims(brain_green)
        return brain_green, dims
    
    elif channel == 'red':
        # try to open motcorr_red, else make it (and make green)
        try:
            print('Trying to load red motion-corrected brain.')
            sys.stdout.flush()
            brain_file = os.path.join(directory, 'motcorr_red.nii')
            brain_red = load_numpy_brain(brain_file)
            print('Loaded red motion-corrected brain.')
            sys.stdout.flush()
        except:
            print('Failed to load red motion-corrected brain.')
            print('Trying to load functional brain.')
            sys.stdout.flush()
            brain_file = folder + '/functional.nii'
            brain_green = load_numpy_brain(brain_file, channel='green')
            brain_red = load_numpy_brain(brain_file, channel='red')
            print('Loaded green and red functional brains.')
            sys.stdout.flush()

            ### Perform motion correction ###
            print('Performing motion-correction.')
            sys.stdout.flush()
            brain_red, brain_green = motion_correction(brain_master=brain_red, brain_slave=brain_green, folder=folder, subfolder=subfolder)
            
        dims = get_dims(brain_red)
        return brain_red, dims
    
    elif channel == None:
        # try to open motcorr, else make it
        try:
            print('Trying to load motion-corrected brain.')
            sys.stdout.flush()
            brain_file = folder + '/motcorr.nii'
            brain = load_numpy_brain(brain_file)
            print('Loaded motion-corrected brain.')
            sys.stdout.flush()
        except:
            # next, try loading the functional brain and perform motion correction
            print('Failed to load motion-corrected brain.')
            print('Trying to load functional brain.')
            sys.stdout.flush()
            brain_file = folder + '/functional.nii'
            brain_green = load_numpy_brain(brain_file, channel='green')
            print('Loaded functional brain.')
            sys.stdout.flush()

            ### Perform motion correction ###
            print('Performing motion-correction.')
            sys.stdout.flush()
            brain_green = motion_correction(brain_master=brain_green, folder=folder, subfolder=subfolder)

        dims = get_dims(brain)
        return brain, dims
    
    else:
        raise Exception('Invalid channel type.')
