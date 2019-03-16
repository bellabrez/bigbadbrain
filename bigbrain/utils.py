import sys
import smtplib
import re
from email.mime.text import MIMEText
from time import time
from functools import wraps

sys.path.insert(0, '/home/users/brezovec/projects/lysis/')
from bruker import *

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
        duration = (end-start)/60
        #save_duration(name=f.__name__, duration=duration)
        print(' {} Elapsed time: {}{}'.format(f.__name__,duration,'min'))
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
    print('Loading timestamps.')
    sys.stdout.flush()
    # load from h5py if it exists, otherwise load from xml and create h5py
    try:
        timestamps = bruker_timestamps_import(folder, file, False)
    except:
        timestamps = bruker_timestamps_import(folder, file, True)
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
