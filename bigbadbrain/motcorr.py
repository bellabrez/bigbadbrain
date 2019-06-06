import numpy as np
import os
import sys
import psutil
from time import time
import matplotlib.pyplot as plt
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore")

from bigbadbrain.brain import get_resolution, get_dims, load_numpy_brain, make_meanbrain
from bigbadbrain.utils import timing

import platform
if platform.system() != 'Windows':
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

    with stderr_redirected(): # to prevent dumb itk gaussian error bullshit infinite printing
        motCorr_vol = ants.registration(fixed, moving_vol, type_of_transform='SyN')

    return motCorr_vol

def split_if_too_big(f):
    def wrapper(*args, **kwargs):
        # If x/y is too big, need to do 1st and 2nd half separately
        dims = get_dims(kwargs['brain_master'])
        if dims['x'] > 200:
            print('Brain too big to motcorr at once - will do it in two parts.')
            sys.stdout.flush()

            middle_volume = int(dims['t']/2)

            print('Starting first half.')
            sys.stdout.flush()
            kwargs['start_volume'] = 0
            kwargs['end_volume'] = middle_volume
            kwargs['suffix'] = '_first_half'
            result = f(*args, **kwargs)
                
            print('Starting second half.')
            sys.stdout.flush()
            kwargs['start_volume'] = middle_volume
            kwargs['end_volume'] = dims['t']
            kwargs['suffix'] = '_second_half'
            result = f(*args, **kwargs)

        else:
            result = f(*args, **kwargs)
    return wrapper

@timing
def motion_correction(brain_master,
                      brain_slave,
                      directory,
                      motcorr_directory,
                      meanbrain=None,
                      start_volume=None,
                      end_volume=None,
                      suffix=''):
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
    
    # Make mean brain if not supplied
    if meanbrain is None:
        meanbrain = ants.from_numpy(make_meanbrain(brain_master))

    dims = get_dims(brain_master)

    # Align each time volume to the meanbrain
    motCorr_brain_master = []
    motCorr_brain_slave = []
    transforms = []
    durations = []
    sys.stdout.flush()

    if start_volume is None:
        start_volume = 0

    if end_volume is None:
        end_volume = dims['t']

    for i in range(start_volume, end_volume):
        #sys.stdout.write('\r')
        #sys.stdout.flush()
        print('Aligning brain volume {} of {}.'.format(i+1, dims['t']), end=' ')
        #memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
        #print('Current memory usage: {:.2f}GB'.format(memory_usage))
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
        
        durations.append(time()-t0)

        print('Average Duration {:.1f}s'.format(np.mean(durations)))
        sys.stdout.flush()

    # Save motcorr brains
    save_motCorr_brain(motCorr_brain_master, motcorr_directory, suffix='red'+suffix)
    save_motCorr_brain(motCorr_brain_slave, motcorr_directory, suffix='green'+suffix)

    save_transform_files(transforms, motcorr_directory, suffix=suffix)
    #save_motion_figure(transform_matrix, directory, motcorr_directory)

@timing
def save_transform_files(transforms, motcorr_directory, suffix):
# Organize mat transform file
    transform_matrix = []
    for i, transform in enumerate(transforms):
        for x in transform:
            if '.mat' in x:
                temp = ants.read_transform(x)
                transform_matrix.append(temp.parameters)
    transform_matrix = np.array(transform_matrix)

    # Save mat transform file
    save_file = os.path.join(motcorr_directory, 'motcorr_params{}'.format(suffix))
    np.save(save_file,transform_matrix)

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

@contextmanager
def stderr_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stderr.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stderr(to):
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stderr(to=old_stderr) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different