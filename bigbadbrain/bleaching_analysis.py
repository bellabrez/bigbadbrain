import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_triangle
from bigbadbrain.utils import timing
import bigbadbrain as bbb

def perform_bleaching_analysis(expt_folder):
    brain_path = os.path.join(expt_folder, 'imaging', 'functional_channel_2.nii')
    brain = bbb.load_numpy_brain(brain_path)
    save_path = os.path.join(expt_folder, 'imaging')

    plt.rcParams.update({'font.size': 24})

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
    plt.title('Bleaching')

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

    bins = np.ndarray.tolist(np.arange(0,260,5))
    fig = plt.figure(figsize=(10,10))
    num_steps = 5
    frame_jump_size = 200
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
    plt.title("Bleaching Histogram")

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
    plt.title('Bleaching')

    fname = os.path.join(save_path, 'bleaching_2.png')
    plt.savefig(fname,dpi=100,bbox_inches='tight')

    #################
    ### Save data ###
    #################

    output_data = {'percent_signal_lost': percent_signal_lost,
               'bleaching_slope': linear_fit[0],
               'threshold': threshold,
               'percent_above_thresh': percent_above_thresh,
               'intensity_histogram_over_time': bin_save}

    save_file = os.path.join(save_path, 'bleaching_analysis.npy')
    np.save(save_file, output_data)