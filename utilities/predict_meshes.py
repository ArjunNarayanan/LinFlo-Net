import torch
import os
import argparse
import yaml
import sys

sys.path.append(os.getcwd())
from src.template import Template
from src.dataset import ImageSegmentationMeshDataset
import vtk_utils.vtk_utils as vtu


def get_deformed_vertices(predictor, image, template_verts):
    image = image.unsqueeze(0)
    template_verts = template_verts.unsqueeze(0)

    with torch.no_grad():
        deformed_verts = predictor(image, template_verts)
    deformed_verts = deformed_verts.squeeze(0)

    return deformed_verts


def write_all_predictions(predictor, dataset, template, output_dir):
    assert os.path.isdir(output_dir)

    template_verts = template.verts_packed()
    for (idx, sample) in enumerate(dataset):
        filename = dataset.get_file_name(idx)
        print("Processing file : ", filename)
        image = sample["image"]
        deformed_verts = get_deformed_vertices(predictor, image, template_verts)
        template = template.update_packed(deformed_verts)

        vtk_mesh = template.to_vtk_mesh()
        out_fn = os.path.join(output_dir, filename + ".vtp")
        vtu.write_vtk_polydata(vtk_mesh, out_fn)


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
    output_dir = config["files"]["out_dir"]
    make_output_dir(output_dir)

    checkpoint_fn = config["files"]["model"]
    checkpoint = torch.load(checkpoint_fn, map_location=torch.device("cpu"))
    net = checkpoint["model"]

    template_fn = config["files"]["template"]
    template = Template.from_vtk(template_fn)

    dataset_fn = config["files"]["root_dir"]
    dataset = ImageSegmentationMeshDataset(dataset_fn)

    write_all_predictions(net, dataset, template, output_dir)
