import numpy as np
import os
import sys
import h5py

def load_visual_stimuli_data(vision_path):
    stimuli = np.load(os.path.join(vision_path, 'stimuli_master.npy'))

    # remove grey_stimuli
    stimuli = [stim[0] for i,stim in enumerate(stimuli) if stimuli[i,0]['name'] != 'Grey']

    # get unique stimuli
    unique_stimuli = [dict(y) for y in set(tuple(x.items()) for x in stimuli)]

    return stimuli, unique_stimuli

def load_photodiode(vision_path):
    # Try to load from h5py file
    try:
        t, pd1, pd2 = load_h5py_pd_data(vision_path)
        
    # First convert from csv to h5py, then load h5py
    except:
        pd_csv_to_h5py(vision_path,'photodiode.csv')
        t, pd1, pd2 = load_h5py_pd_data(vision_path)
    return t, pd1, pd2

def pd_csv_to_h5py(folder, file):
    print('loading raw photodiode data... ',end='')
    #load raw data from csv
    load_file = os.path.join(folder, file)
    temp = np.genfromtxt(load_file, delimiter=',',skip_header=1)
    t = temp[:,0]
    pd1 = temp[:,1]
    pd2 = temp[:,2]
    print('done')

    #save as h5py file
    print('saving photodiode data as h5py file...',end='')
    save_file = os.path.join(folder, 'photodiode.h5')
    with h5py.File(save_file, 'w') as hf:
        hf.create_dataset('time',  data=t)
        hf.create_dataset('pd1',  data=pd1)
        hf.create_dataset('pd2',  data=pd2)
    print('done')

def load_h5py_pd_data(folder):
    print('loading photodiode data... ',end='')
    #load from h5py file
    load_file = os.path.join(folder, 'photodiode.h5')
    with h5py.File(load_file, 'r') as hf:
        t = hf['time'][:]
        pd1 = hf['pd1'][:]
        pd2 = hf['pd2'][:]
    print('done')
    return t, pd1, pd2

def get_pd_thresh(pd):
    _, bins, _ = plt.hist(pd,bins=5)
    plt.close()
    threshold = (bins[0] + bins[-1]) / 2
    return threshold

def parse_stim_starts_photodiode(pd, stimuli):
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