import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nibabel as nib
from time import time
from xml.etree import ElementTree as ET
import os
import pandas as pd
import sys
import scipy
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy import signal
from scipy.ndimage.interpolation import shift
from numpy.linalg import *
from scipy.linalg import toeplitz
import scipy.linalg as sl
from sklearn.cluster import KMeans
import skimage
from sklearn.linear_model import LassoLarsIC
import smtplib
from email.mime.text import MIMEText
import re
import traceback
from IPython.core.magic import register_cell_magic
from functools import wraps

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants

sys.path.insert(0, '/home/users/brezovec/projects/lysis/')
from bruker import *

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

def correlate_brain(brain, to_cor):
    try:
        brain_array = brain.numpy()
        y = np.shape(brain_array)[0]
        x = np.shape(brain_array)[1]
        z = np.shape(brain_array)[2]
        t = np.shape(brain_array)[3]
        print('y: {}, x: {}, z: {}, t: {}'.format(y, x, z, t))

        # first, flatten x and y
        print('flattening...')
        brain_for_cor = np.reshape(brain_array,(y*x,z,t))
        print('done flattening.')
        print('correlating...')
        cors = []
        print('Z-slice: ', end='')
        for z_slice in range(z):
            if z_slice == z:
                print(z_slice, '.')
            else:
                print('{}, '.format(z_slice), end='')
            cors.append(np.corrcoef(brain_for_cor[:,z_slice,:], to_cor[:,z_slice])[-1,:])
        print('done correlating.')
        print('final touches...')
        cors = np.asarray(cors)

        # remove very last cor entry since this is self correlation
        cors = cors[:,:-1]

        #reshape back to correct brain shape
        cor_brain = np.reshape(cors,((z,y,x)))
        print('done.')
    except:
        print('try failed')
        cor_brain = None
    return cor_brain

def calc_event_triggered_delays(numpy_brain,
                                movement_times,
                                timestamps,
                                search_before=2000,
                                search_after=2000):
    
    useful_frames = []

    # For each movement time, find brain slices that occur near in time
    for i, movement_time in enumerate(movement_times):
        print('{} of {}. '.format(i+1, len(movement_times)), end='')
        # Define what time range to look across
        search_start = movement_time - search_before
        search_end = movement_time + search_after

        # Look at each z-slice
        for z in range(len(numpy_brain[0,0,:,0])):

            # Get times of the current slice
            slice_times = timestamps[:,z]

            # Find slices near current movement time
            for f, slice_time in enumerate(slice_times):
                if search_start <= slice_time <= search_end:

                    # Calculate delay
                    delay = slice_time - movement_time

                    # Save the index of the found slice, along with it's delay relative to movement
                    useful_frames.append({'slice': z, 'frame': f, 'delay': delay})
    return useful_frames

def slice_moving_avg(voxel_slices, delays, start=-4000, stop=4000, step=100):
    
    means = []
    centers = []
    
    for window in range(int((stop - start) / step)):
        
        window_start = start + window * step
        window_end = window_start + step

        above = np.asarray([delay > window_start for delay in delays])
        below = np.asarray([delay < window_end for delay in delays])
        indicies = np.where(above & below)[0]
        
        values = [voxel_slices[index,:,:] for index in indicies]
        mean = np.mean(values, axis = 0)
        means.append(mean)
        
    return means

def signal_metric(numpy_brain, useful_frames, metric='variance'):

    brain_triggered = []
    mean_signal = []
    
    mean_brain = np.mean(numpy_brain, axis = 3)

    for z in range(len(numpy_brain[0,0,:,0])):
        print(z)

        voxel_slices = []
        delays = []

        for frame in useful_frames:
            if frame['slice'] == z:

                voxel_slice = numpy_brain[:,:,z,frame['frame']]
                voxel_slices.append(voxel_slice)

                delay = frame['delay']
                delays.append(delay)

        means = slice_moving_avg(np.asarray(voxel_slices), delays)
        means_smoothed = scipy.ndimage.filters.gaussian_filter1d(means, sigma=1, axis=0)
        mean_signal.append(means_smoothed)

        if metric == 'df':
            minimums = np.amin(means_smoothed, axis=0)
            maximums = np.amax(means_smoothed, axis=0)
            value = (maximums - minimums) #/ minimums
        elif metric == 'variance':
            value = np.var(means_smoothed, axis=0)
        elif metric == 'd_from_mean':
            minimums = np.amin(means, axis=0) # note, not using smoothed
            minimums = mean_brain[:,:,z] - minimums
            maximums = np.amax(means, axis=0) # note, not using smoothed
            maximums = maximums - mean_brain[:,:,z]
            value = np.maximum(minimums, maximums)
        elif metric == 'integrated_f':
            value = np.sum(np.abs(means), axis = 0)
            
        brain_triggered.append(value)
    
    brain_triggered = np.asarray(brain_triggered)
    
    return brain_triggered, mean_signal

def single_moving_avg(voxels, delays, start=-2000, stop=2000, step=100):
    means = []
    centers = []
    for window in range(int((stop - start) / step)):
        window_start = start + window * step
        window_end = window_start + step
        center = window_start + (step / 2)
        centers.append(center)
        above = np.asarray([delay > window_start for delay in delays])
        below = np.asarray([delay < window_end for delay in delays])
        indicies = np.where(above & below)[0]
        mean = np.mean([voxels[index] for index in indicies])
        means.append(mean)
    return means, centers

def send_email(subject='', message=''):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("python.notific@gmail.com", "9!tTT77x!ma8cGy")

    msg = MIMEText(message)
    msg['Subject'] = subject

    to = "brezovec@stanford.edu"
    server.sendmail(to, to, msg.as_string())
    server.quit()

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        duration = (end-start)/60
        save_duration(name=f.__name__, duration=duration)
        print('Elapsed time: {}{}'.format(duration,'min'))
        sys.stdout.flush()
        return result
    return wrapper

def save_duration(name, duration):
    function_durations.append({'name': name, 'duration': duration})

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(x):
    x.sort(key=alphanum_key)
    
def tryint(s):
    try:
        return int(s)
    except:
        return s

@timing
def load_fictrac(root_path, fly_folder):
    print('Loading fictrac.')
    sys.stdout.flush()
    with open(os.path.join(root_path, fly_folder, 'fictrac.dat'),'r') as f:
                df = pd.DataFrame(l.rstrip().split() for l in f)

                # Name columns
                df = df.rename(index=str, columns={0: 'frameCounter',
                                               1: 'dRotCamX',
                                               2: 'dRotCamY',
                                               3: 'dRotCamZ',
                                               4: 'dRotScore',
                                               5: 'dRotLabX',
                                               6: 'dRotLabY',
                                               7: 'dRotLabZ',
                                               8: 'AbsRotCamX',
                                               9: 'AbsRotCamY',
                                               10: 'AbsRotCamZ',
                                               11: 'AbsRotLabX',
                                               12: 'AbsRotLabY',
                                               13: 'AbsRotLabZ',
                                               14: 'positionX',
                                               15: 'positionY',
                                               16: 'heading',
                                               17: 'runningDir',
                                               18: 'speed',
                                               19: 'integratedX',
                                               20: 'integratedY',
                                               21: 'timeStamp',
                                               22: 'sequence'})

                # Remove commas
                for column in df.columns.values[:-1]:
                    df[column] = [float(x[:-1]) for x in df[column]]

                fictrac_data = df
                
    # sanity check for extremely high speed (fictrac failure)
    speed = np.asarray(fictrac_data['speed'])
    max_speed = np.max(speed)
    if max_speed > 10:
        raise Exception('Fictrac ball tracking failed (reporting impossibly high speed).')
    return fictrac_data

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
def motion_correction(brain_master=None, brain_slave=None, folder=None):
    send_email('started motion correction')
    # old: numpy_brain, folder
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
        if i == 10:
            send_email('Duration of 1 vol: {:.1f}s'.format(time()-t0))

    # Save motcorr brain(s)
    print('Saving brain...', end='')
    sys.stdout.flush()
    t = time()
    brain_master_motCorr = save_motCorr_brain(motCorr_brain_master, folder, suffix='red')
    
    if brain_slave is not None:
        brain_slave_motCorr = save_motCorr_brain(motCorr_brain_slave, folder, suffix='green')
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
    save_file = os.path.join(folder, 'motcorr_params')
    np.save(save_file,transform_matrix)

    # Get voxel resolution for figure
    print('Getting voxel resolution.')
    sys.stdout.flush()
    file = os.path.join(folder, 'functional.xml')
    x_res, y_res, z_res = get_resolution(file)

    # Save figure of motion over time
    print('Saving motion correction figure.')
    sys.stdout.flush()
    save_file = os.path.join(folder, 'motion_correction.png')
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
def load_timestamps(folder, file='functional.xml'):
    print('Loading timestamps.')
    sys.stdout.flush()
    # load from h5py if it exists, otherwise load from xml and create h5py
    try:
        timestamps = bruker_timestamps_import(folder, file, False)
    except:
        timestamps = bruker_timestamps_import(folder, file, True)
    return timestamps
        
@timing
def prep_fictrac(fictrac, timestamps, fps, dur):
    print('Preping fictrac.')
    sys.stdout.flush()
    camera_rate = 1/fps * 1000 # camera frame rate in ms
    raw_fictrac_times = np.arange(0,dur,camera_rate)
    
    # Cut off any extra frames (only happened with brain 4)
    fictrac = fictrac[:90000]
    
    # Smooth
    fictrac_smoothed = scipy.ndimage.filters.gaussian_filter(np.asarray(fictrac['speed']),sigma=3)
    
    # Interpolate
    # Warning: interp1d set to fill in out of bounds times
    fictrac_interp_temp = interp1d(raw_fictrac_times, fictrac_smoothed, bounds_error = False)
    fictrac_interp = fictrac_interp_temp(timestamps)
    
    # Replace Nans with zeros (for later code)
    np.nan_to_num(fictrac_interp, copy=False);
    
    return fictrac_interp

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
            brain_red, brain_green = motion_correction(brain_master=brain_red, brain_slave=brain_green, folder=directory)
        
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
            brain_red, brain_green = motion_correction(brain_master=brain_red, brain_slave=brain_green, folder=directory)
            
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
            brain_green = motion_correction(brain_master=brain_green, folder=folder)

        dims = get_dims(brain)
        return brain, dims
    
    else:
        raise Exception('Invalid channel type.')

def get_dims(brain):
    y = brain.shape[0]
    x = brain.shape[1]
    z = brain.shape[2]
    t = brain.shape[3]
    return {'x': x, 'y': y, 'z': z, 't': t}

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

@timing
def fit_glm(brain, dims, fictrac, beta_len):
    print('Fit GLM.')
    sys.stdout.flush()
    middle = int((beta_len - 1) / 2)
    models = []
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
                #models.append(model)
                scores.append(model.score(X,Y))
    scores = np.reshape(scores, (dims['z'], dims['x'], dims['y']))
    return scores

def save_glm_map(vol, folder, channel):
    print('Saving glm vol.')
    sys.stdout.flush()
    file = 'multivariate_analysis_' + channel + '.nii'
    save_file = os.path.join(folder, file)
    brain_to_save = np.swapaxes(vol, 0, 2)
    ants.image_write(ants.from_numpy(brain_to_save), save_file)

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

desired_flies = [21] # 1 index
fly_folders = [fly_folders[i-1] for i in desired_flies]
flies = [flies[i-1] for i in desired_flies]
print(fly_folders)
sys.stdout.flush()

beta_len = 21 #MUST BE ODD
fps = 50 #of fictrac camera
dur = 30 * 60 * 1000 # experiment duration in ms
vols_to_clip = 200

for fly_idx, fly in enumerate(flies):
    
    ### Send email and define folder path ###
    function_durations = []
    print('Starting analysis of {}.'.format(fly_folders[fly_idx]))
    sys.stdout.flush()
    send_email('Starting {} ({} of {}).'.format(fly_folders[fly_idx], fly_idx+1, len(flies)), 'wow')
    folder = root_path + fly_folders[fly_idx]

    ### Load timestamps ###
    timestamps = load_timestamps(folder)
    
    ### Load fictrac (and prep) ###
    # add sanity check for failed frames. Simply check if and speed measurements are over 10
    fictrac = load_fictrac(root_path, fly_folders[fly_idx])
    print('About to prep')
    sys.stdout.flush()
    fictrac_interp = prep_fictrac(fictrac, timestamps, fps, dur)
    print('Just preped')
    sys.stdout.flush()
    
    # remove first bit of data since it often has some weirdness
    timestamps = timestamps[vols_to_clip:,:]
    fictrac_interp = fictrac_interp[vols_to_clip:,:]
    
    send_email('loaded timestamps and fictrac', 'wow')
    
    ############# Do remaining analysis on both brain channels ###############
    channels = ['green', 'red']
    for channel in channels:
        ### Load brain ###
        brain, dims = get_motcorr_brain(folder, channel=channel)

        # remove first bit of data since it often has some weirdness
        brain = brain[:,:,:,vols_to_clip:]
        dims['t'] = brain.shape[3]

        ### Bleaching correction (per voxel) ###
        brain = bleaching_correction(brain)

        ### Z-score brain ###
        brain = z_score_brain(brain)

        ### Fit GLM ###
        scores = fit_glm(brain, dims, fictrac_interp, beta_len)

        ### Save brain ###
        save_glm_map(scores, folder, channel)

        ### Send email ###

        # Prep timing string
        func_str = ''
        for func in function_durations:
            func_str += '{} ===== {:.2f} min\n'.format(func['name'], func['duration'])
        print(func_str)
        sys.stdout.flush()

        #send_email('Success {} ({} of {}):{} {} channel.'.format(fly_folders[fly_idx], fly_idx+1, len(flies)), func_str, channel)
        print('Reached END.')
        sys.stdout.flush()