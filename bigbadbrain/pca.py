import numpy as np
import os
import sys
import bigbadbrain as bbb
from sklearn.decomposition import PCA

def main(expt_folder):

    brain_path = os.path.join(directory, 'brain_zscored_green.nii')
    brain = bbb.load_numpy_brain(brain_path)
    dims = bbb.get_dims(brain)
    print('brain shape is x,y,z,t {}'.format(brain.shape))

    t0 = time()
    X = brain[:,:,:,:].reshape(-1,brain.shape[-1]).T
    brain = None
    print('X is time by voxels {}'.format(X.shape))
    print('Reshape duration: {}'.format(time()-t0))

    t0 = time()
    pca = PCA().fit(X)
    pca_scores = pca.components_
    print('Scores is PC by voxel {}'.format(pca_scores.shape))
    pca_loadings = pca.transform(X)
    print('Loadings is time by PC {}'.format(pca_loadings.shape))
    pca_spatial = np.reshape(pca_scores, (-1,dims['x'],dims['y'],dims['z']))
    print('Spatial is {}'.format(pca_spatial.shape))
    print('PCA duration: {}'.format(time()-t0))

    save_file = os.path.join(directory, 'pca', 'scores_(spatial).npy')
    np.save(save_file, pca_spatial)
    save_file = os.path.join(directory, 'pca', 'loadings_(temporal).npy')
    np.save(save_file, pca_loadings)
    print('Saved PCA!')

if __name__ == '__main__':
    main(sys.argv[1])