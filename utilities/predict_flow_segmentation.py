import torch
import os
import argparse
import yaml
import sys
import pandas as pd

sys.path.append(os.getcwd())
import src.io_utils as io
import vtk_utils.vtk_utils as vtu


def get_segmentation(predictor, image):
    image = image.unsqueeze(0)

    with torch.no_grad():
        segmentation = predictor.get_segmentation(image)

    segmentation = segmentation.squeeze(0)

    return segmentation


def write_all_predictions(predictor, input_dir, output_dir):
    assert os.path.isdir(output_dir)
    assert os.path.isdir(input_dir)

    index_fn = os.path.join(input_dir, "index.csv")
    index = pd.read_csv(index_fn)

    for idx in range(len(index)):
        filename = index.iloc[idx, 0]
        print("Processing file : ", filename)

        image_fn = os.path.join(input_dir, "vtk_image", filename + ".vti")
        vtk_img = vtu.load_vtk_image(image_fn)
        torch_img = io.vtk_image_to_torch(vtk_img).unsqueeze(0)

        segmentation = get_segmentation(predictor, torch_img)
        segmentation = segmentation.argmax(dim=0)

        vtk_segmentation = io.torch_to_vtk_image(segmentation, vtk_img)
        out_fn = os.path.join(output_dir, filename + ".vti")
        vtu.write_vtk_image(vtk_segmentation, out_fn)


def get_config(config_fn):
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def make_output_dir(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        print("WARNING : OUTPUT FOLDER EXISTS, DATA WILL BE OVERWRITTEN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model predictions and write mesh files")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config = get_config(args.config)
    output_dir = config["output_dir"]
    make_output_dir(output_dir)

    checkpoint_fn = config["checkpoint"]
    checkpoint = torch.load(checkpoint_fn, map_location=torch.device("cpu"))
    net = checkpoint["model"]

    input_dir = config["input_dir"]
    index_fn = os.path.join(input_dir, "index.csv")
    assert os.path.isfile(index_fn)

    image_dir = os.path.join(input_dir, "vtk_image")
    assert os.path.isdir(image_dir)

    write_all_predictions(net, input_dir, output_dir)
