import os
import sys
import bigbadbrain as bbb

def main(args):
    directory = args[0]
    channel = args[1]
    behavior = args[2]
    sign = args[3]

    beta_len = 21 #MUST BE ODD
    fps = 50 #of fictrac camera
    dur = 30 * 60 * 1000 # experiment duration in ms
    sigma = 3

    bbb.announce_start(directory)
    timestamps = bbb.load_timestamps(os.path.join(directory, 'imaging'))
    fictrac = bbb.load_fictrac(os.path.join(directory, 'fictrac'))
    brain = bbb.get_z_brain(directory, channel)
    dims = bbb.get_dims(brain)


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

if __name__ == '__main__':
    main(sys.argv[1:])