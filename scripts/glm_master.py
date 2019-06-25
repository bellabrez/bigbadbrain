import os
import numpy as np
import argparse

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
    if args.behavior:
        jobs = [' '.join(['sbatch', 'behavior_glm.sh', expt, channel, behavior, sign])
                for expt in expt_folders
                for channel in args.channels
                for behavior in args.b_behaviors
                for sign in args.b_signs]
        #[print(job) for job in jobs]
        [os.system(job) for job in jobs]

    if args.visual:
        for expt in expt_folders:
            stimuli, unique_stimuli = load_visual_stimuli_data(os.path.join(expt, 'visual'))
            jobs = [' '.join(['sbatch', 'visual_glm.sh', expt, channel, str(args.v_bin_size),\
                    str(args.v_pre_dur), str(args.v_post_dur), str(stim_index)])
                    for channel in args.channels
                    for stim_index in range(len(unique_stimuli))]
            #[print(job) for job in jobs]
            [os.system(job) for job in jobs]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--flies', nargs='+', type=str, required=True,
        help='Supply fly numbers separated by spaces. Can also supply specific experiments with flynum_exptnum, ie 33_0. Will run on all\
        expts if no expt specified.')
    parser.add_argument('-c', '--channels', nargs='+', choices=['g', 'r'], help='which brain channels to use', default=['g'], type=str)

    parser.add_argument('-v', '--visual', action='store_true', help='')
    parser.add_argument('--v_bin_size', default=100, type=str, help='')
    parser.add_argument('--v_pre_dur', default=500, type=str, help='')
    parser.add_argument('--v_post_dur', default=1500, type=str, help='')

    parser.add_argument('-b', '--behavior', action='store_true', help='')
    parser.add_argument('--b_signs', nargs='+', choices=['original', 'abs', 'df_abs','plus', 'minus', 'df'], default = ['original'], type=str)
    parser.add_argument('--b_behaviors', nargs='+', choices=['dRotLabY', 'dRotLabZ', 'dRotLabX', 'speed'], default = ['dRotLabY'], type=str)
    
    args = parser.parse_args()
    main(args)
