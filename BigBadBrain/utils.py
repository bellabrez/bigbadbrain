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
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("python.notific@gmail.com", "9!tTT77x!ma8cGy")

    msg = MIMEText(message)
    msg['Subject'] = subject

    to = "brezovec@stanford.edu"
    server.sendmail(to, to, msg.as_string())
    server.quit()

def save_duration(name, duration):
    function_durations.append({'name': name, 'duration': duration})

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
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

        #save_duration(name=f.__name__, duration=duration)
        print('Duration: {:.2f} {}'.format(duration,suffix))
        sys.stdout.flush()
        return result
    return wrapper

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
def load_timestamps(folder, file='functional.xml'):
    print('\n~~ Loading Timestamps ~~')
    sys.stdout.flush()

    # load from h5py if it exists, otherwise load from xml and create h5py
    try:
        print('Trying to load timestamp data from hdf5 file.')
        with h5py.File(os.path.join(folder, 'timestamps.h5'), 'r') as hf:
            timestamps = hf['timestamps'][:]

    except:
        print('Failed. Extracting frame timestamps from bruker xml file.')
        xml_file = os.path.join(folder, file)
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
        with h5py.File(os.path.join(folder, 'timestamps.h5'), 'w') as hf:
            hf.create_dataset("timestamps", data=timestamps)
    
    print('Success.')
    return timestamps

def get_fly_folders(root_path, desired_flies):
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
