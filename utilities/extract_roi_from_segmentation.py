import SimpleITK as sitk
import numpy as np
import glob
import os
import pandas as pd
import argparse


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
    lower = np.array([x0, y0, z0])
    upper = np.array([x1, y1, z1])
    return lower, upper


def get_bounds(seg, background_id=0):
    seg_arr = sitk.GetArrayFromImage(seg)
    mask = seg_arr != background_id
    lower, upper = get_bounds_on_mask(mask)
    return lower, upper


def get_perturbed_bounds_and_size(seg_fn, margin=range(20, 31)):
    seg = sitk.ReadImage(seg_fn)
    lower, upper = get_bounds(seg)

    lower = lower - np.random.choice(margin, 3)
    lower = np.maximum(lower, np.zeros(3, dtype=int))

    size = np.array(seg.GetSize())
    upper = upper + np.random.choice(margin, 3)
    upper = np.minimum(upper, size)

    return lower, upper, size


def get_all_bounds(filenames):
    bounds = []
    for fn in filenames:
        seg = sitk.ReadImage(fn)
        lower, upper = get_bounds(seg)
        _bounds = np.array([lower, upper]).T.ravel()
        bounds.append(_bounds)
    bounds = np.array(bounds)
    return bounds


def get_all_perturbed_bounds_and_size(filenames):
    bounds = []
    sizes = []
    for fn in filenames:
        lower, upper, size = get_perturbed_bounds_and_size(fn)
        _bounds = np.array([lower, upper]).T.ravel()
        bounds.append(_bounds)
        sizes.append(size)
    bounds = np.array(bounds)
    sizes = np.array(sizes)
    return bounds, sizes


def get_max_size(filename):
    seg = sitk.ReadImage(filename)
    size = np.array(seg.GetSize())
    return size


def get_all_sizes(filenames):
    sizes = []
    for fn in filenames:
        size = get_max_size(fn)
        sizes.append(size)
    sizes = np.array(sizes)
    return sizes


def get_labels(seg_fn):
    seg = sitk.ReadImage(seg_fn)
    seg_arr = sitk.GetArrayFromImage(seg)
    labels = np.unique(seg_arr)
    return labels


def get_all_labels(filenames):
    labels = []
    for fn in filenames:
        _labels = get_labels(fn)
        labels.append(_labels)

    labels = np.array(labels)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute region of interest from segmentations")
    parser.add_argument("-f", help="folder to process")
    parser.add_argument("-o", default="crop_bounds.csv", help="output csv file")
    args = parser.parse_args()

    base_dir = args.f
    root_dir = os.path.join(base_dir, "label")
    assert os.path.isdir(root_dir)
    filepaths = glob.glob(os.path.join(root_dir, "*.nii.gz"))
    filepaths.sort()
    print("Found ", len(filepaths), " files")

    bounds, sizes = get_all_perturbed_bounds_and_size(filepaths)

    dx = bounds[:, 1] - bounds[:, 0]
    dy = bounds[:, 3] - bounds[:, 2]
    dz = bounds[:, 5] - bounds[:, 4]
    deltas = np.stack([dx, dy, dz], axis=1)

    filenames = [os.path.basename(fn) for fn in filepaths]
    index = [fn.split(".")[0] for fn in filenames]

    df = pd.DataFrame({"sample_name": index})
    df[["X0", "X1", "Y0", "Y1", "Z0", "Z1"]] = bounds
    df[["dX", "dY", "dZ"]] = deltas
    df[["Xm", "Ym", "Zm"]] = sizes

    cropped_volume = dx * dy * dz
    original_volume = sizes.prod(axis=1)
    vol_ratio = cropped_volume/original_volume
    df["volume_ratio"] = vol_ratio

    df = df[["sample_name", "X0", "X1", "dX", "Xm", "Y0", "Y1", "dY", "Ym", "Z0", "Z1", "dZ", "Zm", "volume_ratio"]]

    out_dir = base_dir
    out_file = args.o
    out_file_path = os.path.join(out_dir, out_file)
    print("Writing bounds data to ", out_file_path)
    df.to_csv(out_file_path, index=False)