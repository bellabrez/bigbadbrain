import numpy as np
from time import time
import os
import sys
import scipy
import copy
import argparse

sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants
import bigbadbrain as bbb

def main(args):
    # this file will use os.system to start separate glm jobs
    # need to use argparse variables
    # get list of experiment folders / brain channels
    root_path = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'

    expt_folders = []
    for item in args.flies:
        # Handle case where a specific experiment is provided
        if '_' in item:
            flynum = item.split('_')[0]
            exptnum = item.split('_')[1]
            expt_folder = os.path.join(root_path, 'fly_' + flynum, 'func_' + exptnum)
            if os.path.isdir(expt_folder):
                expt_folders.append(expt_folder)
            else:
                print('Experiment folder does not exist: {}'.format(expt_folder))
        # Handle case where an entire fly is provided
        else:
            fly_folder = os.path.join(root_path, 'fly_' + item)
            if not os.path.isdir(fly_folder):
                print('Fly folder does not exist: {}'.format(fly_folder))
            folders = [os.path.join(root_path, fly_folder, x) for x in os.listdir(fly_folder) if 'func' in x]
            expt_folders.extend(folders)
    print('Proceeding with these experiment folders: {}'.format(expt_folders))

    # Launch glms
    if behavior:
        jobs = [['sbatch', 'behavior_glm.sh', expt, channel, behavior, sign]
                for expt in expt_folders
                for channel in args.channels
                for behavior in args.b_behaviors
                for sign in args.b_signs]
        jobs = [' '.join(x) for x in jobs]
        [os.system(job) for job in jobs]


        print(jobs)


        for expt in expt_folders
            for channel in args.channels:
                for behavior in args.b_behaviors:
                    for sign in args.b_signs:
                        os.system('sbatch behavior_glm.sh ' + expt + ' ' + channel + ' ' + behavior + ' ' + sign)







    ##########################
    ### What flies to run? ###
    ##########################
    sys.stdout.flush()
    
    desired_flies = [33, 36] # 1 index
    folders = bbb.get_fly_folders(root_path, desired_flies)

    ######################################
    ### What brain areas and channels? ###
    ######################################
    channels = ['green']
    brain_regions = ['func_0', 'func_1'] # if not nested, put ''

    ################################
    ### Perform visual analysis? ###
    ################################
    visual = True
    bin_size = 100 #in ms
    pre_dur = 500 #in ms
    post_dur = 1500 #in ms

    ####################################
    ### Perform behavioral analysis? ###
    ####################################
    behavior = True
    signs =  ['df_abs'] # abs, plus, minus, df, or None
    behaviors = ['dRotLabY', 'dRotLabZ'] #'dRotLabX', 'dRotLabY', 'speed'
    fictrac_sigmas = [3]
    beta_len = 21 #MUST BE ODD
    fps = 50 #of fictrac camera
    dur = 30 * 60 * 1000 # experiment duration in ms

    #############
    ### START ###
    #############
    for fly_idx, folder in enumerate(folders):
        for brain_region in brain_regions:

            directory = os.path.join(folder, brain_region)
            bbb.announce_start(directory, fly_idx, folders)
            timestamps = bbb.load_timestamps(os.path.join(directory, 'imaging'))

            if behavior:
                fictrac = bbb.load_fictrac(os.path.join(directory, 'fictrac'))
            if visual:
                unique_stimuli = bbb.get_stimuli(os.path.join(directory, 'visual'))
            
            for channel in channels:
                brain = bbb.get_z_brain(directory, channel)
                dims = bbb.get_dims(brain)

                if behavior:
                    for behavior in behaviors:
                        for sigma in fictrac_sigmas:
                            for sign in signs:
                                ### Prep given behavior ###
                                fictrac_interp = bbb.interpolate_fictrac(fictrac,
                                                                         timestamps,
                                                                         fps,
                                                                         dur,
                                                                         behavior=behavior,
                                                                         sigma=sigma,
                                                                         sign=sign)

                                ### Fit GLM ###
                                scores, betas = bbb.fit_glm(brain, fictrac_interp, beta_len)

                                ### Save brain ###
                                behavior_info = {'behavior': behavior,
                                                 'sigma': sigma,
                                                 'sign': sign}
                                metadict = bbb.make_glm_meta_dict('behavior', channel, behavior_info)
                                bbb.save_glm_map(scores, betas, directory, metadict)

                if visual:
                    for stimulus in unique_stimuli:

                        ### Fit GLM ###
                        scores, betas = bbb.fit_visual_glm(brain, stimulus, timestamps, bin_size, pre_dur, post_dur)

                        ### Save brain ###
                        stimulus_no_times = copy.deepcopy(stimulus)
                        stimulus_no_times.pop('times', None)
                        metadict = bbb.make_glm_meta_dict('visual', channel, stimulus_no_times)
                        bbb.save_glm_map(scores, betas, directory, metadict)

                if behavior and visual:
                    for stimulus in unique_stimuli:
                    
                        ### Create behavior STA plot ###
                        bbb.create_stim_triggered_behavior_plot(fictrac, stimulus, directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--flies', nargs='+', type=str, required=True,
        help='Supply fly numbers separated by spaces. Can also supply specific experiments with flynum_exptnum, ie 33_0. Will run on all\
        expts if no expt specified.')
    parser.add_argument('-c', '--channels', nargs='+', choices=['g', 'r'], help='which brain channels to use', default=['g'], type=str)

    parser.add_argument('-v', '--visual', action='store_true', help='')
    parser.add_argument('--v_bin_size', default=100, type=int, help='')
    parser.add_argument('--v_pre_dur', default=500, type=int, help='')
    parser.add_argument('--v_post_dur', default=1500, type=int, help='')

    parser.add_argument('-b', '--behavior', action='store_true', help='')
    parser.add_argument('--b_signs', nargs='+', choices=['original', 'abs', 'df_abs','plus', 'minus', 'df'], default = ['original'], type=str)
    parser.add_argument('--b_behaviors', nargs='+', choices=['dRotLabY', 'dRotLabZ', 'dRotLabX', 'speed'], default = ['dRotLabY'], type=str)
    
    args = parser.parse_args()
    main(args)