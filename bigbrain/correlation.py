import numpy as np

def correlate_brain(brain, to_cor):
    try:
        brain_array = brain.numpy()
        y = np.shape(brain_array)[0]
        x = np.shape(brain_array)[1]
        z = np.shape(brain_array)[2]
        t = np.shape(brain_array)[3]
        print('y: {}, x: {}, z: {}, t: {}'.format(y, x, z, t))

        # first, flatten x and y
        print('flattening...')
        brain_for_cor = np.reshape(brain_array,(y*x,z,t))
        print('done flattening.')
        print('correlating...')
        cors = []
        print('Z-slice: ', end='')
        for z_slice in range(z):
            if z_slice == z:
                print(z_slice, '.')
            else:
                print('{}, '.format(z_slice), end='')
            cors.append(np.corrcoef(brain_for_cor[:,z_slice,:], to_cor[:,z_slice])[-1,:])
        print('done correlating.')
        print('final touches...')
        cors = np.asarray(cors)

        # remove very last cor entry since this is self correlation
        cors = cors[:,:-1]

        #reshape back to correct brain shape
        cor_brain = np.reshape(cors,((z,y,x)))
        print('done.')
    except:
        print('try failed')
        cor_brain = None
    return cor_brain