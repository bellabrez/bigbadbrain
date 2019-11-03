import os
import sys
import numpy as np
from time import strftime
from shutil import copyfile
from xml.etree import ElementTree as ET
from lxml import etree, objectify
import bigbadbrain as bbb
import pandas as pd
import json
import bigbadbrain as bbb
import psutil

def main():
    print('Performing X on all flies.')
    root_directory = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/'
    fly_folders = [os.path.join(root_directory,x) for x in os.listdir(root_directory) if 'fly' in x]
    bbb.sort_nicely(fly_folders)
    fly_folders = fly_folders[::-1]

    # TO SELECT ONLY A SUBSET OF FLIES

    #fly_folders = [os.path.join(root_directory, 'fly_68'), os.path.join(root_directory, 'fly_69')]
    fly_folders = []
    #for i in np.arange(3,5,7,19,21,48,51,54,68):
    for i in [3,5,7,19,21,48,51,54,68]:
        fly_folders.append(os.path.join(root_directory, 'fly_{}'.format(i)))
    #fly_folders = [os.path.join(root_directory, 'fly_1')]

    for fly in fly_folders:
        expt_folders = []
        expt_folders = [os.path.join(fly,x) for x in os.listdir(fly) if 'func' in x]
        if len(expt_folders) > 0:
            for expt_folder in expt_folders:
                memory_usage = psutil.Process(os.getpid()).memory_info().rss*10**-9
                print('Current memory usage in loop: {:.2f}GB'.format(memory_usage))
                sys.stdout.flush()
                print('Performing X on: {}'.format(expt_folder))
                try:
                    #_ = bbb.perform_bleaching_analysis(expt_folder)
                    os.system("sbatch run_X_single.sh {}".format(expt_folder))

                except:
                    print('Try block failed for {}'.format(expt_folder))

if __name__ == '__main__':
    main()
