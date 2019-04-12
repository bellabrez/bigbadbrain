import sys
import smtplib
import re
import os
import h5py
import math
from email.mime.text import MIMEText
from time import time
from functools import wraps
import numpy as np
import nibabel as nib
from scipy.ndimage import imread
from xml.etree import ElementTree as ET

def send_email(subject='', message=''):
    """ Sends emails!

    Parameters
    ----------
    subject: email subject heading (str)
    message: body of text (str)

    Returns
    -------
    Nothing. """

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("python.notific@gmail.com", "9!tTT77x!ma8cGy")

    msg = MIMEText(message)
    msg['Subject'] = subject

    to = "brezovec@stanford.edu"
    server.sendmail(to, to, msg.as_string())
    server.quit()

def timing(f):
    """ Wrapper function to time how long functions take (and print function name). """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        print('\n~~ {} ~~'.format(f.__name__))
        sys.stdout.flush()
        result = f(*args, **kwargs)
        end = time()
        duration = end-start

        # Make units nice (originally in seconds)
        if duration < 1:
            duration = duration * 1000
            suffix = 'ms'
        elif duration < 60:
            duration = duration
            suffix = 'sec'
        elif duration < 3600:
            duration = duration / 60
            suffix = 'min'
        else:
            duration = duration / 3600
            suffix = 'hr'

        print('Done. Duration: {:.2f} {}'.format(duration,suffix))
        sys.stdout.flush()
        return result
    return wrapper

def alphanum_key(s):
    """ Tries to change strs to ints. """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(x):
    """

    Parameters
    ----------

    Returns
    -------

    """
    x.sort(key=alphanum_key)
    
def tryint(s):
    """ Tries to change a single str to an int. """

    try:
        return int(s)
    except:
        return s

@timing
def load_timestamps(directory, file='functional.xml'):
    """ Parses a Bruker xml file to get the times of each frame, or loads h5py file if it exists.

    First tries to load from 'timestamps.h5' (h5py file). If this file doesn't exist
    it will load and parse the Bruker xml file, and save the h5py file for quick loading in the future.

    Parameters
    ----------
    directory: full directory that contains xml file (str).
    file: Defaults to 'functional.xml'

    Returns
    -------
    timestamps: [t,z] numpy array of times (in ms) of Bruker imaging frames.

    """
    try:
        print('Trying to load timestamp data from hdf5 file.')
        with h5py.File(os.path.join(directory, 'timestamps.h5'), 'r') as hf:
            timestamps = hf['timestamps'][:]

    except:
        print('Failed. Extracting frame timestamps from bruker xml file.')
        xml_file = os.path.join(directory, file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        timestamps = []
        
        sequences = root.findall('Sequence')
        for sequence in sequences:
            frames = sequence.findall('Frame')
            for frame in frames:
                filename = frame.findall('File')[0].get('filename')
                time = float(frame.get('relativeTime'))
                timestamps.append(time)
        timestamps = np.multiply(timestamps, 1000)

        if len(sequences) > 1:
            timestamps = np.reshape(timestamps, (len(sequences), len(frames)))
        else:
            timestamps = np.reshape(timestamps, (len(frames), len(sequences)))

        ### Save h5py file ###
        with h5py.File(os.path.join(directory, 'timestamps.h5'), 'w') as hf:
            hf.create_dataset("timestamps", data=timestamps)
    
    print('Success.')
    return timestamps

def get_fly_folders(root_path, desired_flies):
    """ Will create an ordered list of all subdirectories listed in desired_flies.

    Subfolders must contain 'fly' in the name, followed by a fly number (i.e. fly_1).

    Parameters
    ----------
    root_path: directory containing fly folders
    desired_flies: list of ints (1 index) of desired flies.

    Returns
    -------
    folders: list of full path to desired fly folders """

    fly_folders = sorted(os.listdir(root_path))
    fly_folders = [x for x in fly_folders if 'fly' in x]
    sort_nicely(fly_folders)
    fly_folders = [fly_folders[i-1] for i in desired_flies]

    print('fly folders: {}'.format(fly_folders))
    sys.stdout.flush()

    folders = []
    for fly_folder in fly_folders:
        folders.append(os.path.join(root_path + fly_folder))

    return folders

def fft_signal(signal, sampling_rate, duration):
    """ Performs FFT on a signal.

    Parameters
    ----------
    signal: 1D numpy array
    sampling_rate: in Hz (I think... check)
    duration: int (in sec?). Can probably calulate this instead of requiring.

    Returns
    -------
    y: signal
    Y: fft
    t: times """
    
    Fs = sampling_rate
    y = signal
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,duration,Ts) # time vector
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]

def announce_start(directory, fly_idx, folders):
    ### Send email and define folder path ###
    print('\n~~~~ Starting analysis of {} ~~~~'.format(directory))
    sys.stdout.flush()
    send_email('Starting {} ({} of {}).'.format(directory, fly_idx+1, len(folders)), 'wow')