import sys
import os
import pickle
import numpy as np
import torch
import argparse
import pandas as pd
import yaml

sys.path.append(os.getcwd())
import src.io_utils as io


def get_image(img_fn):
    img = io.read_image(img_fn)
    assert img.ndim == 3
    img = img.unsqueeze(0)
    return img


def read_segmentation(seg_fn):
    img = io.read_image(seg_fn)
    assert img.ndim == 3
    return img


def relabel_segmentation(seg, classes):
    for (idx, c) in enumerate(classes):
        seg[seg == c] = idx

    return seg


def get_segmentation(seg_fn, seg_labels):
    num_classes = len(seg_labels)

    seg = read_segmentation(seg_fn)
    seg_classes = np.unique(seg)
    assert len(seg_labels) == len(seg_classes)

    if all(seg_classes == seg_labels):
        seg = relabel_segmentation(seg, seg_labels)
    else:
        assert all(seg_classes == range(num_classes))

    seg = seg.to(torch.long)
    return seg


def pickle_image_segmentation_mesh(im_fn, seg_fn, mesh_fn, seg_labels, out_fn):
    assert os.path.isfile(im_fn) and os.path.isfile(seg_fn) and os.path.isfile(mesh_fn)
    img = get_image(im_fn)
    seg = get_segmentation(seg_fn, seg_labels)
    meshes = io.pytorch3d_meshes_from_vtk(mesh_fn)

    data = {"image": img, "segmentation": seg, "meshes": meshes}
    pickle.dump(data, open(out_fn, "wb"))


def pickle_all_data(root_dir, filename_index, outdir, seg_labels):
    for idx in range(len(filename_index)):
        filename = filename_index.iloc[idx, 0]
        print("Pickling file : ", filename)

        image_fn = os.path.join(root_dir, "vtk_image", filename + ".vti")
        mesh_fn = os.path.join(root_dir, "vtk_mesh", filename + ".vtp")
        seg_fn = os.path.join(root_dir, "vtk_segmentation", filename + ".vti")
        out_fn = os.path.join(outdir, filename + ".pkl")

        pickle_image_segmentation_mesh(image_fn, seg_fn, mesh_fn, seg_labels, out_fn)


def check_input_folders(root_dir):
    assert os.path.isdir(os.path.join(root_dir, "vtk_image")), "Missing vtk_image folder"
    assert os.path.isdir(os.path.join(root_dir, "vtk_segmentation")), "Missing vtk_segmentation folder"
    assert os.path.isdir(os.path.join(root_dir, "vtk_mesh")), "Missing vtk_mesh folder"
    assert os.path.isfile(os.path.join(root_dir, "index.csv")), "Missing index file"


def make_output_folders(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    else:
        print("\n\nWARNING - output folder already exists, data will be overwritten\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle image segmentation and mesh data")
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
