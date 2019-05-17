import os
import sys
import numpy as np
import scipy
import psutil
import nibabel as nib
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from BigBadBrain.utils import timing

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

@timing
def load_numpy_brain(file, channel=None, flip_z=False):
    """ Loads nifti file into numpy array.

    Uses nibabel to load nifti file and converts to a float64 numpy array.
    Throughout this package, nifti files and numpy arrays are in order y,x,z,t (note y, then x).
    This is to match nifti conventions. This can be annoying when using plt.imshow, since the brain
    will be plotted sideways. So, you should use the function view_brain, which will swap the 
    axes for viewing.

    Parameters
    ----------
    file: string of full path to nifti (.nii) file.
    channel: None by default. Can specify 'red' (axis=4 is 0) or 'green' (axis=4 is 1).
    flip_z: False by default. Will flip z axes (axis=2).

    Returns
    -------
    brain: [x,y,z,t,(c)] numpy array. """

    # If no channel specified, load all channels, else load specified channel
    brain = nib.load(file).get_data()
    if channel == 'red':
        brain = brain[:,:,:,:,0] # for red brain.
    if channel  == 'green':
        brain = brain[:,:,:,:,1] # for green brain.
    #brain = np.swapaxes(brain, 0, 1)
    if flip_z is True:
        brain = np.flip(brain, 2)
    brain = np.squeeze(brain)
    brain = np.asarray(brain, 'float64')
    #brain = ants.from_numpy(brain)
    return brain

@timing
def save_brain(file, brain):
    """ Saves numpy array as nifti file using antspy.

    Parameters
    ----------
    file: string of full path to nifit (.nii) file to save.
    brain: numpy array to save.

    Returns
    -------
    Nothing. """

    ants.image_write(ants.from_numpy(brain), file)
    memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
    print('Current memory usage: {:.2f}GB'.format(memory_usage))
    sys.stdout.flush()

def view_brain(brain, slice_idx=None):
    """ Uses matplotlib imshow to view a brain slice.

    Parameters
    ----------
    brain: 3D numpy array with axis order y,x,z.
    slice_idx: index of slice to view (0 index). If slice not provided, will show the middle slice.

    Returns
    -------
    Nothing. """

    # Get middle slice if none provided.
    if slice_idx is None:
        slice_idx = int(brain.shape[-1]/2)

    plt.figure(figsize=(10,10))
    plt.imshow(np.swapaxes(brain,0,1)[:,:,slice_idx])
    plt.show()

@timing
def load_brains_from_dir(directory):
    """ Will load all nifti files in a given directory using load_numpy_brain.

    Does not search subdirectories.

    Parameters
    ----------
    directory: string of directory from which to load all brains

    Returns
    -------
    brains: list of numpy arrays """

    brain_files = sorted(os.listdir(directory))
    full_brain_files = [directory+brain_file for brain_file in brain_files]
    brains = [load_numpy_brain(brain) for brain in full_brain_files]
    return brains

def rotate_brain(brain):
    """ Swaps x and y axes (axes 0 and 1) using np.swapaxes.

    Will try to swap axes, otherwise will return None
    (allows None to be passed to function without failing).

    Parameters
    ----------
    brain: numpy array or antspy object

    Returns
    -------
    brain: antspy object or None.

    """
    try:
        if type(brain) is not np.ndarray:
            brain = brain.numpy()
        brain = np.swapaxes(brain, 0, 1)
        brain = ants.from_numpy(brain)
    except:
        brain = None
        print('Returning None.')
    return brain

def set_resolution(brain, xml_file):
    """ Sets the spacing (resolution) of an antspy object based on parsing of Bruker xml_file.

    In units of microns, assumes x,y,z order. x/y order doesn't matter if equal resolution.

    Parameters
    ----------
    brain: antspy object.
    xml_file: string to full path of Bruker xml file.

    Returns
    -------
    Nothing. Modifies in place. """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    statevalues = root.findall('PVStateShard')[0].findall('PVStateValue')
    for statevalue in statevalues:
        key = statevalue.get('key')
        if key == 'micronsPerPixel':
            indices = statevalue.findall('IndexedValue')
            for index in indices:
                axis = index.get('index')
                if axis == 'XAxis':
                    x = float(index.get('value'))
                elif axis == 'YAxis':
                    y = float(index.get('value'))
                elif axis == 'ZAxis':
                    z = float(index.get('value'))
                else:
                    print('Error')
    try:
        brain.set_spacing([x,y,z])
    except:
        print('Failed')
        
def get_resolution(xml_file):
    """ Gets the x,y,z resolution of a Bruker recording.

    Units of microns.

    Parameters
    ----------
    xml_file: string to full path of Bruker xml file.

    Returns
    -------
    x: float of x resolution
    y: float of y resolution
    z: float of z resolution """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    statevalues = root.findall('PVStateShard')[0].findall('PVStateValue')
    for statevalue in statevalues:
        key = statevalue.get('key')
        if key == 'micronsPerPixel':
            indices = statevalue.findall('IndexedValue')
            for index in indices:
                axis = index.get('index')
                if axis == 'XAxis':
                    x = float(index.get('value'))
                elif axis == 'YAxis':
                    y = float(index.get('value'))
                elif axis == 'ZAxis':
                    z = float(index.get('value'))
                else:
                    print('Error')
    return x, y, z

@timing
def bleaching_correction(brain,sigma=200):
    """ Subtracts slow brain trends over time.

    Subtracts each voxel's slow-pass truncated gaussian filter from itself.
    The slow-pass filtering will be different for different speeds of aquisition since
    sigma is in units of indicies, not time. Not worrying about this for now since
    my imaging is all similar aquisition rates (2-3Hz).

    Parameters
    ----------
    brain: numpy array. Time must be axis=-1.
    sigma: sigma of gaussian filter. sigma=200 is default. At 2Hz aquisition, this is a
    200*(1/2Hz)*2 = 200sec or ~3min window of smoothing.

    Returns
    -------
    brain: original numpy array with slow trends subtracted."""

    print('brain_shape: {}'.format(np.shape(brain)))
    sys.stdout.flush()

    smoothed = scipy.ndimage.gaussian_filter1d(brain,sigma=sigma,axis=-1,truncate=1)
    brain = brain - smoothed
    return brain

@timing
def z_score_brain(brain):
    """ Subtracts means and divides by stddev for each voxel independently.

    Parameters
    ----------
    brain: 4D numpy array. Time must by last axis.

    Returns
    -------
    brain: 4D z-scored numpy array.

    """

    brain_mean  = np.mean(brain, axis=3)
    brain_std = np.std(brain, axis=3)
    brain = (brain - brain_mean[:,:,:,None]) / brain_std[:,:,:,None]
    return brain

def get_dims(brain):
    """ Creates a dict of numpy array length of each axis.
    Parameters
    ----------
    brain: 4D numpy array.  Assumes axis order is y,x,z,t.

    Returns
    -------
    Dictionary of {'y': y, 'x': x, 'z': z, 't': t}.

    """
    y = brain.shape[0]
    x = brain.shape[1]
    z = brain.shape[2]
    t = brain.shape[3]
    return {'y': y, 'x': x, 'z': z, 't': t}

@timing
def make_meanbrain(brain):
    return np.mean(brain, axis=-1)