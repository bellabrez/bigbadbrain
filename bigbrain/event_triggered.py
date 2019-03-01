import numpy
import scipy

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
