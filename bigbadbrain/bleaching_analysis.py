import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_triangle
from bigbadbrain.utils import timing
import bigbadbrain as bbb
import psutil

def main(expt_folder):

    print('Expt folder is: {}'.format(expt_folder))

    # Get fly num and expt num for graph titles
    expt_num = os.path.split(expt_folder)[-1]
    fly_num = os.path.split(os.path.split(expt_folder)[0])[-1]
    title = 'Bleaching, {}, {}'.format(fly_num,expt_num)

    memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
    print('Starting memory usage: {:.2f}GB'.format(memory_usage))
    sys.stdout.flush()

    brain = None
    try:
        brain_path = os.path.join(expt_folder, 'imaging', 'functional_channel_2.nii')
        brain = bbb.load_numpy_brain(brain_path)
    except:
        print('Failed to load functional_channel_2.nii')
        try:
            brain_path = os.path.join(expt_folder, 'imaging', 'functional.nii')
            brain = bbb.load_numpy_brain(brain_path)
            brain = brain[:,:,:,:,-1] # Take only green channel
            print('Successfully loaded functional.nii')
        except:
            print('Failed to load functional.nii')
            try: 
                brain_path = os.path.join(expt_folder, 'imaging', 'stitched_brain_green.nii')
                brain = bbb.load_numpy_brain(brain_path)
                print('Successfully loaded stitched_brain_green.nii')
            except:
                print('FAILED to load any brain.')
                return

    save_path = os.path.join(expt_folder, 'imaging')

    plt.rcParams.update({'font.size': 24})

    # Convert to 8-bit if it is 13-bit
    if np.max(brain[:,:,:,:10]) > 256: # Just get max from first 10 frames
        bit_depth = 13
        brain = np.multiply(brain,(2**8/2**13))
    else:
        bit_depth = 8

    print('Detected bit depth: {}'.format(bit_depth))

    ##############################
    ### Output Bleaching Curve ###
    ##############################

    fig = plt.figure(figsize=(10,10))
    data_mean = np.mean(brain,axis=(0,1,2))
    xs = np.arange(brain.shape[-1])
    plt.plot(data_mean,color=np.divide((51, 212, 155),255))
    linear_fit = np.polyfit(xs, data_mean, 1)
    plt.plot(np.poly1d(linear_fit)(xs),color='k',linewidth=3,linestyle='--')
    plt.text

    plt.xlabel('Frame Num')
    plt.ylabel('Avg signal (8bit)')
    plt.title(title)

    percent_signal_lost = linear_fit[0]*brain.shape[-1]/linear_fit[1]*-100
    print('percent_signal_lost: {}'.format(percent_signal_lost))
    plt.text(0.8,0.9,
             'Lost {}%'.format(int(percent_signal_lost)),
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes)

    fname = os.path.join(save_path, 'bleaching_0.png')
    plt.savefig(fname,dpi=100,bbox_inches='tight')

    ##################################
    ### Output Bleaching Histogram ###
    ##################################

    threshold = threshold_triangle(np.ndarray.flatten(brain[:,:,:,::100]))

    # The number of histograms that will be created
    # ie, how many partitions to split timeseries into
    num_steps = 5

    # Under sampling each partition
    # Will run faster with bigger numbers
    frame_jump_size = 200

    # For final histogram
    #num_bins = 50

    #bins = np.ndarray.tolist(np.arange(0,2**bit_depth,int(2**bit_depth/num_bins)))
    bins = np.ndarray.tolist(np.arange(0,255,5))
    fig = plt.figure(figsize=(10,10))

    step = int(brain.shape[-1]/num_steps)

    start_color = (4,217,255)
    end_color = (170,35,255)
    color_increment = np.subtract(end_color, start_color)/num_steps
    bin_save = []

    for i in range(num_steps):
        start = step*i
        end = start+step
        vector = np.ndarray.flatten(brain[:,:,:,start:end:frame_jump_size])
        weights = np.ones_like(vector)/float(len(vector))
        if i == 0:
            label = 'Beginning'
        elif i == num_steps-1:
            label = 'End'
        else:
            label = None
        color = tuple(start_color + color_increment * i)
        color = np.divide(color,255)
        binned,throwaway,throwaway2=plt.hist(vector,
                                             bins,
                                             weights=weights*100,
                                             log=True,
                                             label=label,
                                             stacked=True,
                                             histtype='step',
                                             linewidth=3,
                                             color=color)
        bin_save.append(binned)
    plt.axvline(threshold,color='k',linewidth=3,linestyle='--')
    plt.xlabel("Intensity, 8 bits")
    plt.ylabel("Frequency, %")
    plt.legend(loc=1)
    plt.title(title)

    fname = os.path.join(save_path, 'bleaching_1.png')
    plt.savefig(fname,dpi=100,bbox_inches='tight')

    ################################
    ### Output % Below Threshold ###
    ################################

    step = 10
    percent_above_thresh = []
    for i in range(0,brain.shape[-1],step):
        data = brain[:,:,:,i]
        percent_above_thresh.append(len(data[data>threshold])/len(np.ndarray.flatten(data)))

    fig = plt.figure(figsize=(10,10))
    xs = np.arange(0,brain.shape[-1],step)
    plt.plot(xs, np.multiply(percent_above_thresh,100),color=np.divide((51, 212, 155),255),linewidth=3)
    plt.ylabel('Voxels above thresh (%)')
    plt.xlabel('Frame Num')
    plt.title(title)

    fname = os.path.join(save_path, 'bleaching_2.png')
    plt.savefig(fname,dpi=100,bbox_inches='tight')

    #################
    ### Save data ###
    #################

    output_data = {'percent_signal_lost': percent_signal_lost,
                   'bleaching_slope': linear_fit[0],
                   'threshold': threshold,
                   'percent_above_thresh': percent_above_thresh,
                   'intensity_histogram_over_time': bin_save,
                   'bleaching': data_mean}

    save_file = os.path.join(save_path, 'bleaching_analysis.npy')
    np.save(save_file, output_data)

    memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
    print('Ending memory usage: {:.2f}GB'.format(memory_usage))
    sys.stdout.flush()

    return None

if __name__ == '__main__':
    main(sys.argv[1])
