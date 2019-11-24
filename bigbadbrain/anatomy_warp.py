import numpy as np
import os
import sys
import bigbadbrain as bbb
import time
sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
import ants
import scipy

def main(directory):

    anat_folder = directory

    ##### Load template #####
    template_file = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190224_anatomy_central/meanbrain1/JFRCtemplate2010.nii'
    template = bbb.load_numpy_brain(template_file)
    template = template[:,:,::-1] # Flip Z-axis
    template = template[200:800,:,:] # Cut off optic lobes
    template = ants.from_numpy(template)
    ants.set_spacing(template, (0.622, 0.622, 0.622)) # Set resolution

    ##### Load individual #####
    fly_num = os.path.split(os.path.split(anat_folder)[0])[-1]
    anat_num = os.path.split(anat_folder)[-1]
    this_anat = '{},{}'.format(fly_num, anat_num)
    if this_anat == 'fly_12,anat_0':
        anatomy = bbb.load_numpy_brain(os.path.join(anat_folder, 'anat_mean.nii'))
        print('Loaded SPECIAL {}'.format(this_anat))
    else:
        try:
            anatomy = bbb.load_numpy_brain(os.path.join(anat_folder, 'anatomy_channel_1.nii'))
            print('Loaded {}'.format(this_anat))
        except:
            print('failed to load anatomy_channel_1.nii: {}'.format(this_anat))
            try:
                anatomy = bbb.load_numpy_brain(os.path.join(anat_folder, 'anatomy.nii'))
            except:
                print('failed to load anatomy.nii: {}'.format(this_anat))

    # Rotate some brains
    if this_anat in ['fly_7,anat_0', 'fly_2,anat_0', 'fly_1,anat_0']:
        anatomy = np.swapaxes(anatomy,0,1)

    anatomy_xml = os.path.join(anat_folder, 'anatomy.xml')
    anatomy_ants = ants.from_numpy(anatomy)
    anatomy_ants.set_spacing(bbb.get_resolution(anatomy_xml))

    ##### Warp #####
    grad_steps = 0.2
    flow_sigma = 10
    total_sigma = 0
    aff_sampling = 32
    syn_sampling = 256
    verbose = True
    grad_steps = 0.2

    print('Working on {}'.format(this_anat))
    t0 = time.time()
    slave2master = ants.registration(template,
                                     anatomy_ants,
                                     type_of_transform='SyN',
                                     syn_sampling=syn_sampling,
                                     flow_sigma=flow_sigma,
                                     total_sigma=total_sigma,
                                     grad_steps=grad_steps)

    MI_after = ants.image_mutual_information(template,slave2master['warpedmovout'])
    print('COMPLETE. Duration: {:0.0f} sec'.format(time.time()-t0))

    ##### Save #####
    save_folder = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset/20191123_meanbrain'
    file = os.path.join(save_folder, '{}_{}.nii'.format(this_anat,MI_after))
    bbb.save_brain(file, slave2master['warpedmovout'].numpy())

if __name__ == '__main__':
    main(sys.argv[1])