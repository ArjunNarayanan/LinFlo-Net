import sys
import os
import pickle
import numpy as np
import torch.nn.functional as func
import argparse
import pandas as pd
import yaml

sys.path.append(os.getcwd())
import src.io_utils as io


def relabel_segmentation(seg, classes):
    seg_classes = np.unique(seg)
    assert len(seg_classes) == len(classes)
    assert all(seg_classes == classes)

    for (idx, c) in enumerate(classes):
        seg[seg == c] = idx

    return seg


def get_segmentation(seg_fn, seg_labels):
    seg = io.read_image(seg_fn)
    assert seg.ndim == 3
    seg = relabel_segmentation(seg, seg_labels)
    return seg


def get_image(img_fn):
    img = io.read_image(img_fn)
    img = img.unsqueeze(0)
    return img


def pickle_image_and_segmentation(im_fn, seg_fn, seg_labels, out_fn):
    assert os.path.isfile(im_fn) and os.path.isfile(seg_fn)
    img = get_image(im_fn)
    seg = get_segmentation(seg_fn, seg_labels)

    data = {"image": img, "segmentation": seg}
    pickle.dump(data, open(out_fn, "wb"))


def pickle_all_data(root_dir, filename_index, outdir, seg_labels):
    for idx in range(len(filename_index)):
        filename = filename_index.iloc[idx, 0]
        print("Pickling filename : ", filename)

        im_fn = os.path.join(root_dir, "vtk_image", filename + ".vti")
        seg_fn = os.path.join(root_dir, "vtk_segmentation", filename + ".vti")
        out_fn = os.path.join(outdir, filename + ".pkl")

        pickle_image_and_segmentation(im_fn, seg_fn, seg_labels, out_fn)


def check_input_folders(root_dir):
    assert os.path.isdir(os.path.join(root_dir, "vtk_image")), "Missing vtk_image folder"
    assert os.path.isdir(os.path.join(root_dir, "vtk_segmentation")), "Missing vtk_segmentation folder"
    assert os.path.isfile(os.path.join(root_dir, "index.csv")), "Missing index file"


def make_output_folders(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        print("WARNING - output folder already exists, data will be overwritten")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle image and segmentation data")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)


    infolder = config["input_folder"]
    outfolder = config["output_folder"]
    seg_labels = config["seg_labels"]

    check_input_folders(infolder)
    make_output_folders(outfolder)

    index_file = os.path.join(infolder, "index.csv")
    index = pd.read_csv(index_file)

    pickle_all_data(infolder, index, outfolder, seg_labels)
