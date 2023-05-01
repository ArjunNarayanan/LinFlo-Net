import torch
from src.template import Template
from src.linear_transform import *
import os
import vtk_utils.vtk_utils as vtu
import pickle
import numpy as np


def write_deformation_history(sample_name, num_samples=20):
    sample_path = os.path.join(dataset_folder, sample_name + ".pkl")
    assert os.path.isfile(sample_path)
    sample = pickle.load(open(sample_path, "rb"))
    image = sample["image"]

    template_verts = template.verts_packed().unsqueeze(0)
    factors = np.linspace(0, 1, num_samples)

    print("Writing deformation history with : ", num_samples, " time-steps")

    for (idx, multiplication_factor) in enumerate(factors):
        with torch.no_grad():
            deformed_verts = model(image, template_verts, multiplication_factor)

        deformed_template = template.update_packed(deformed_verts.squeeze(0))
        vtk_template = deformed_template.to_vtk_mesh()

        out_filename = sample_name + "_{:02d}".format(idx) + ".vtp"
        out_filepath = os.path.join(output_folder, out_filename)

        vtu.write_vtk_polydata(vtk_template, out_filepath)


output_root_dir = "output/WholeHeartData/trained_models/mr/linear_transform/meshes/incremental"
dataset_folder = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/mr/processed/indexed/data"
model_file = "output/WholeHeartData/trained_models/mr/linear_transform/best_model.pth"
template_file = "data/template/highres_template.vtp"
sample_name = "sample05"

data = torch.load(model_file, map_location=torch.device("cpu"))
model = data["model"]

template = Template.from_vtk(template_file)

output_folder = os.path.join(output_root_dir, sample_name)
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

write_deformation_history(sample_name)
