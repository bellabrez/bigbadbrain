import numpy as np
import os
import sys
import h5py
import scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from BigBadBrain.utils import create_bins

def get_stimuli(directory):
    """ Uses photodiode recording and outputs of visual_stimulation package to get a dictionary
    of presented stimuli and their onset times.

    Parses the photodiode recording (currently only using one photodiode) to find onsets of stimuli
    (based on the photodiode going from off to on at a rate slower than frame flipping). Then gets a list of
    all stimuli presented and in what order from 'stimuli_master.npy'. Makes sure the lengths match. Then finds
    unique stimuli, and for each unique stimulus saves all the times that stimuli was presented based on the photodiode.

    Parameters
    ----------
    directory: full path to fly folder. Must contain 'visual' subfolder with required files
    ('photodiode.csv' from Bruker and 'stimuli_master.npy' from visual_stimulation)

    Returns
    -------
    unique_stimuli: dictionary of presented stimuli and their onset times. """

    vision_path = os.path.join(directory, 'visual')
    t,pd1,pd2 = load_photodiode(vision_path)
    stimuli, unique_stimuli = load_visual_stimuli_data(vision_path)
    stimuli_starts = parse_stim_starts_photodiode(pd1,stimuli)
    unique_stimuli = set_unique_stimuli_display_times(unique_stimuli, stimuli, stimuli_starts)
    return unique_stimuli

def load_visual_stimuli_data(vision_path):
    """ Gets unique stimuli from 'stimuli_master.npy', and removes 'Grey' stimuli.

    Parameters
    ----------
    vision_path: full path to vision folder

    Returns
    -------
    stimuli: List of all stimuli presented in order
    unique_stimuli: List of unique stimuli

    """
    print('loading visual stimuli data... ',end='')
    stimuli = np.load(os.path.join(vision_path, 'stimuli_master.npy'))

    # remove grey_stimuli
    stimuli = [stim[0] for i,stim in enumerate(stimuli) if stimuli[i,0]['name'] != 'Grey']

    # get unique stimuli
    unique_stimuli = [dict(y) for y in set(tuple(x.items()) for x in stimuli)]

    print('done')

    return stimuli, unique_stimuli

def load_photodiode(vision_path):
    """ Tries to load photodiode data from h5py file, and if it doesn't exist loads from csv file.

    Parameters
    ----------
    vision_path: full path to vision folder

    Returns
    -------
    t: 1D numpy array, times of photodiode measurement (in ms)
    pd1: 1D numpy array, photodiode 1 measurements
    pd2: 1D numpy array, photodiode 1 measurements """

    # Try to load from h5py file
    try:
        t, pd1, pd2 = load_h5py_pd_data(vision_path)
        
    # First convert from csv to h5py, then load h5py
    except:
        pd_csv_to_h5py(vision_path,'photodiode.csv')
        t, pd1, pd2 = load_h5py_pd_data(vision_path)
    return t, pd1, pd2

def pd_csv_to_h5py(directory, file):
    """ Loads photodiode data from csv file and saves to h5py file.

    Parameters
    ----------
    directory: full path to vision folder
    file: csv file

    Returns
    -------
    Nothing. """

    print('loading raw photodiode data... ',end='')
    #load raw data from csv
    load_file = os.path.join(directory, file)
    temp = np.genfromtxt(load_file, delimiter=',',skip_header=1)
    t = temp[:,0]
    pd1 = temp[:,1]
    pd2 = temp[:,2]
    print('done')

    #save as h5py file
    print('saving photodiode data as h5py file...',end='')
    save_file = os.path.join(directory, 'photodiode.h5')
    with h5py.File(save_file, 'w') as hf:
        hf.create_dataset('time',  data=t)
        hf.create_dataset('pd1',  data=pd1)
        hf.create_dataset('pd2',  data=pd2)
    print('done')

def load_h5py_pd_data(directory):
    """ Loads photodiode data from h5py file.

    Parameters
    ----------
    directory: full path to vision folder

    Returns
    -------
    t: 1D numpy array, times of photodiode measurement (in ms)
    pd1: 1D numpy array, photodiode 1 measurements
    pd2: 1D numpy array, photodiode 1 measurements """

    print('loading photodiode data... ',end='')
    #load from h5py file
    load_file = os.path.join(directory, 'photodiode.h5')
    with h5py.File(load_file, 'r') as hf:
        t = hf['time'][:]
        pd1 = hf['pd1'][:]
        pd2 = hf['pd2'][:]
    print('done')
    return t, pd1, pd2

def set_unique_stimuli_display_times(unique_stimuli, stimuli, stimuli_starts):
    """ Gives each unique stimuli a dictionary item of times of onset.
    Parameters
    ----------
    unique_stimuli: dictionary of unique stimuli
    stimuli: list of all stimuli presented, in order.
    stimuli_starts: list of start times of all stimuli, in order.

    Returns
    -------
    unique_stimuli: dictionary of unique stimuli, now containing onset times.


    """
    for unique_stimulus in unique_stimuli:
        unique_stimulus['times'] = [stimuli_starts[i] for i, x in enumerate(stimuli) if x == unique_stimulus]
    return unique_stimuli

def get_pd_thresh(pd):
    """ Finds threshold for ON vs OFF photodiode.

    Parameters
    ----------
    pd: photodiode values across time

    Returns
    -------
    threshold: int, threshold of ON vs OFF for photodiode

    """
    _, bins, _ = plt.hist(pd,bins=5)
    plt.close()
    threshold = (bins[0] + bins[-1]) / 2
    return threshold

def parse_stim_starts_photodiode(pd, stimuli):
    """ Finds all times in the photodiode recording when a stimulus began.

    Parameters
    ----------
    pd: photodiode values across time
    stimuli: all stimuli presented. Used for checking length of stimuli starts to make sure all stimuli were found

    Returns
    -------
    stimuli_starts: list of start times of all stimuli in order (in ms).

    """
    # Get threshold for on/off
    threshold = get_pd_thresh(pd)

    # Find timepoints when the light is off (and convert to ms)
    pd_sampling_to_ms_conversion = 10
    pd_off = np.where(pd<threshold)[0]/pd_sampling_to_ms_conversion

    # Find timepoints of rising edges
    min_gap = 1 # Lets give a 1 ms window for defining edges
    pd_on_edges = pd_off[np.where(np.diff(pd_off)>min_gap)[0]]

    # Remove any very early on edges cause by projector still getting ready etc.
    remove_on_before = 60*1000 # Currently a 1 min minimum
    pd_on_edges = [x for x in pd_on_edges if x > remove_on_before]
    
    # Add on edge at 0
    pd_on_edges = np.insert(pd_on_edges,0,0)

    # Find on_edges that correspond to stimuli presentation
    min_gap = 100 # Edges greater than 100ms apart are counted as stimuli starts
    stimuli_starts = pd_on_edges[np.where(np.diff(pd_on_edges)>min_gap)[0]+1]
    
    # Make sure we have successfully found all stimuli
    if len(stimuli_starts) == len(stimuli):
        print('Successfully parsed all stimuli from photodiode output.')
    else:
        print(len(stimuli_starts))
        print(len(stimuli))
        raise Exception('Failed to successfully parse stimuli from photodiode output')
    return stimuli_starts #in ms

def create_stim_triggered_behavior_plot(fictrac,
                                        stimulus,
                                        folder,
                                        behavior='dRotLabZ',
                                        fps=50,  #in ms
                                        dur=30*60*1000,  #in ms
                                        pre_stim=500,  #in ms
                                        post_stim=1000,  #in ms
                                        sampling_res=20):  #in ms
    
    camera_rate = 1/fps * 1000
    raw_fictrac_times = np.arange(0,dur,camera_rate)

    sphere_radius = 4.5e-3
    fictrac_vector = np.rad2deg(np.asarray(fictrac[behavior])) # now in deg per 20ms
    fictrac_vector = fictrac_vector * 50 # now in deg per sec
    fictrac_smoothed = scipy.ndimage.filters.gaussian_filter(fictrac_vector,sigma=3,truncate=1)
    fictrac_interp = interp1d(raw_fictrac_times, fictrac_smoothed, bounds_error = False)

    behavior_chunks = []
    for time in stimulus['times']:
        times = np.arange(time-pre_stim,time+post_stim+1,sampling_res) #+1 is so last time point is included
        behavior_chunks.append(fictrac_interp(times))

    avg_trace = np.mean(np.asarray(behavior_chunks),axis=0)
    x = np.arange(pre_stim,post_stim+1,sampling_res)

    plt.figure(figsize=(10,10))
    for behavior_chunk in behavior_chunks:
        plt.plot(x,behavior_chunk,color=str(np.random.uniform()))
    plt.ylabel('Z Rotation Velocity, deg/sec')
    plt.xlabel('Time relative to stimulus, ms')
    plt.title(folder)
    plt.legend()

    save_file = os.path.join(folder, 'behavior_STA_{}_angle_{}.png'.format(behavior, stimulus['angle']))
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
