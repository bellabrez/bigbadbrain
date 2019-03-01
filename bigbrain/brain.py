import numpy
import os
import sys
import scipy
import nibabel as nib
from xml.etree import ElementTree as ET

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

def testit():
    print('HEY MAN.')

def load_numpy_brain(file, channel=None, flip=False):
    # If no channel specified, load all channels, else load specified channel
    brain = nib.load(file).get_data()
    if channel == 'red':
        brain = brain[:,:,:,:,0] # for red brain.
    if channel  == 'green':
        brain = brain[:,:,:,:,1] # for green brain.
    #brain = np.swapaxes(brain, 0, 1)
    if flip is True:
        brain = np.flip(brain, 2)
    brain = np.squeeze(brain)
    brain = np.asarray(brain, 'float64')
    #brain = ants.from_numpy(brain)
    return brain

def load_brains_from_dir(folder):
    brain_files = sorted(os.listdir(folder))
    full_brain_files = [folder+brain_file for brain_file in brain_files]
    brains = [load_brain(brain) for brain in full_brain_files]
    return brains

def rotate_brain(brain):
    try:
        if type(brain) is not np.ndarray:
            brain = brain.numpy()
        brain = np.swapaxes(brain, 0, 1)
        brain = ants.from_numpy(brain)
    except:
        brain = None
    return brain

def set_resolution(brain, xml_file):
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
def bleaching_correction(brain):
    print('Bleaching correction.')
    sys.stdout.flush()
    smoothed = scipy.ndimage.gaussian_filter1d(brain,sigma=200,axis=3,truncate=1)
    brain = brain - smoothed
    return brain

@timing
def z_score_brain(brain):
    print('Z-score brain.')
    sys.stdout.flush()
    brain_mean  = np.mean(brain, axis=3)
    brain_std = np.std(brain, axis=3)
    brain = (brain - brain_mean[:,:,:,None]) / brain_std[:,:,:,None]
    return brain

def get_dims(brain):
    y = brain.shape[0]
    x = brain.shape[1]
    z = brain.shape[2]
    t = brain.shape[3]
    return {'x': x, 'y': y, 'z': z, 't': t}