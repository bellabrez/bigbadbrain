import numpy as np
import sys
import os
import scipy
import pandas as pd
from scipy.interpolate import interp1d

from bigbrain.utils import timing

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