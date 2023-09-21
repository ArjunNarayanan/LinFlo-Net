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


def get_perturbed_bounds_and_size(seg_fn, margin=range(30, 50)):
    seg = sitk.ReadImage(seg_fn)
    lower, upper = get_bounds(seg)

    lower = lower - np.random.choice(margin, 3)
    lower = np.maximum(lower, np.zeros(3, dtype=int))

    size = np.array(seg.GetSize())
    upper = upper + np.random.choice(margin, 3)
    upper = np.minimum(upper, size)

    return lower, upper, size


def get_all_bounds(filenames):
    print("Computing image bounds:")
    bounds = []
    for fn in filenames:
        seg = sitk.ReadImage(fn)
        lower, upper = get_bounds(seg)
        _bounds = np.array([lower, upper]).T.ravel()
        bounds.append(_bounds)
    bounds = np.array(bounds)
    return bounds


def get_all_perturbed_bounds_and_size(filenames, lower_margin=30, upper_margin=50):
    print("Computing cropped boundaries:")
    margin = range(lower_margin, upper_margin)
    bounds = []
    sizes = []
    for fn in filenames:
        lower, upper, size = get_perturbed_bounds_and_size(fn, margin)
        _bounds = np.array([lower, upper]).T.ravel()
        bounds.append(_bounds)
        sizes.append(size)
    bounds = np.array(bounds)
    sizes = np.array(sizes)
    return bounds, sizes


def get_all_spacings(filenames):
    spacings = []
    for fn in filenames:
        img = sitk.ReadImage(fn)
        spacings.append(img.GetSpacing())

    spacings = np.array(spacings)
    return spacings


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
    parser.add_argument("-lmargin", default=30, help="minimum margin to add", type=int)
    parser.add_argument("-umargin", default=49, help="maximum margin to add", type=int)
    args = parser.parse_args()

    base_dir = args.f
    root_dir = base_dir
    assert os.path.isdir(root_dir)
    filepaths = glob.glob(os.path.join(root_dir, "*.nii.gz"))
    filepaths.sort()
    print("Found ", len(filepaths), " files")
    lower_margin = args.lmargin
    upper_margin = args.umargin

    perturbed_bounds, sizes = get_all_perturbed_bounds_and_size(filepaths, lower_margin, upper_margin+1)
    bounds = get_all_bounds(filepaths)
    spacings = get_all_spacings(filepaths)

    dx = bounds[:, 1] - bounds[:, 0]
    dy = bounds[:, 3] - bounds[:, 2]
    dz = bounds[:, 5] - bounds[:, 4]

    filenames = [os.path.basename(fn) for fn in filepaths]
    index = [fn.split(".")[0] for fn in filenames]

    df = pd.DataFrame({"sample_name": index})
    df[["X0", "X1", "Y0", "Y1", "Z0", "Z1"]] = perturbed_bounds
    df[["XS", "YS", "ZS"]] = spacings

    cropped_volume = dx * dy * dz
    original_volume = sizes.prod(axis=1)
    vol_ratio = cropped_volume / original_volume
    df["volume_ratio"] = vol_ratio

    df = df[["sample_name", "X0", "X1", "XS", "Y0", "Y1", "YS", "Z0", "Z1", "ZS", "volume_ratio"]]

    out_dir = os.path.dirname(base_dir)
    out_file = args.o
    out_file_path = os.path.join(out_dir, out_file)
    print("Writing bounds data to ", out_file_path)

    out_ext = os.path.splitext(out_file_path)[1]
    if out_ext == ".csv":
        df.to_csv(out_file_path, index=False)
    elif out_ext == ".xlsx":
        df.to_excel(out_file_path, index=False)
    else:
        raise ValueError("Unexpected extension type ", out_ext)
