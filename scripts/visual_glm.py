import os
import sys
import copy
import json
import bigbadbrain as bbb

def main(args):
    directory = args[0]
    channel = args[1]
    bin_size = int(args[2])
    pre_dur = int(args[3])
    post_dur = int(args[4])
    stim_idx = int(args[5])

    #bbb.announce_start(directory)
    timestamps = bbb.load_timestamps(os.path.join(directory, 'imaging'))

    # get stim info
    file = os.path.join(directory, 'visual', 'visual.json')
    with open(file, 'r') as f:  
        stimulus = json.load(f)[stim_idx]
    print('Visual glm says stim_idx is {}'.format(stim_idx))

    brain = bbb.get_z_brain(directory, channel)
    dims = bbb.get_dims(brain)

    ### Fit GLM ###
    scores, betas = bbb.fit_visual_glm(brain, stimulus, timestamps, bin_size, pre_dur, post_dur)

    ### Save brain ###
    stimulus_no_times = copy.deepcopy(stimulus)
    stimulus_no_times.pop('times', None)
    metadict = bbb.make_glm_meta_dict('visual', channel, stimulus_no_times)

    print('Visual glm says stimulus is {}'.format(stimulus_no_times))

    bbb.save_glm_map(scores, betas, directory, metadict)

if __name__ == '__main__':
    main(sys.argv[1:])