import pickle
import copy
import numpy as np


def load_keypoints(keypoints_path):
    with open(keypoints_path, 'rb') as f:
        keypoints = pickle.load(f)
    assert isinstance(keypoints, np.ndarray), "keypoints should be a list of numpy arrays"
    assert keypoints.shape[1] == 17 and keypoints.shape[2] == 3, "keypoints should be a list of numpy arrays with shape [T, 17, 3]"
    return keypoints


def interp_nan_keypoints(kps):
    kps = np.asarray(kps, dtype=float)
    T, J, C = kps.shape
    out = kps.copy()

    for j in range(J):
        for c in range(C):
            ts = np.arange(T)
            y = out[:, j, c]

            mask = ~np.isnan(y)
            if mask.sum() == 0:
                continue
            if mask.sum() == T:
                continue

            ts_valid = ts[mask]
            y_valid = y[mask]
            y_interp = np.interp(ts, ts_valid, y_valid)
            out[:, j, c] = y_interp

    return out


def resample_keypoints_1d(x, T_target):
    T = x.shape[0]
    t_orig = np.arange(T)
    t_new = np.linspace(0, T - 1, T_target)
    return np.interp(t_new, t_orig, x)


def resample_keypoints_2d(kps, T_target):
    T, J, C = kps.shape

    if T == T_target:
        return kps.copy()

    out = np.empty((T_target, J, C), dtype=float)

    for j in range(J):
        for c in range(C):
            out[:, j, c] = resample_keypoints_1d(kps[:, j, c], T_target)

    return out


"""
following functions are adopted from MotionBERT:
https://github.com/Walter0807/MotionBERT/blob/dccfe7b14a7f09aae2e27a16d951cb4c6f6e62f6/lib/data/dataset_action.py
"""

def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam


def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y


"""
adopted from MotionBERT:
https://github.com/Walter0807/MotionBERT/blob/dccfe7b14a7f09aae2e27a16d951cb4c6f6e62f6/lib/utils/utils_data.py
"""

def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2]- [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result
