import SimpleITK as sitk
import numpy as np
import glob
import os


def get_bounds_on_last_axis(mask):
    projected_mask = mask.any(axis=0).any(axis=0)
    indices = projected_mask.nonzero()[0]
    low_index = indices[0]
    up_index = indices[-1]
    return low_index, up_index


def get_bounds_on_mask(mask):
    x0, x1 = get_bounds_on_last_axis(mask)
    mask = mask.transpose([2, 0, 1])
    y0, y1 = get_bounds_on_last_axis(mask)
    mask = mask.transpose([2, 0, 1])
    z0, z1 = get_bounds_on_last_axis(mask)
    lower = [x0, y0, z0]
    upper = [x1, y1, z1]
    return lower, upper


def get_bounds(seg_fn, background_id=0):
    seg = sitk.ReadImage(seg_fn)
    seg_arr = sitk.GetArrayFromImage(seg)
    mask = seg_arr != background_id
    lower, upper = get_bounds_on_mask(mask)
    return lower, upper


def get_all_bounds(filenames):
    bounds = []
    for fn in filenames:
        lower, upper = get_bounds(fn)
        _bounds = np.array([lower, upper]).T.ravel()
        bounds.append(_bounds)
    bounds = np.array(bounds)
    return bounds


# image_fn = "/Users/arjunnarayanan/Documents/Research/Simcardio/Datasets/HeartData-samples/train/mr/image/la_001.nii.gz"
# seg_fn = "/Users/arjunnarayanan/Documents/Research/Simcardio/Datasets/HeartData-samples/train/mr/label/la_001.nii.gz"

root_dir = "/Users/arjunnarayanan/Documents/Research/Simcardio/Datasets/HeartData-samples/train/mr/label"
filepaths = glob.glob(os.path.join(root_dir, "*.nii.gz"))
filepaths.sort()
bounds = get_all_bounds(filepaths)

filenames = [os.path.basename(fn) for fn in filepaths]