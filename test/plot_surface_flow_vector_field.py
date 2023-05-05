import torch
from src.template import Template
import os
import pickle
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk_utils.vtk_utils as vtu
from src.integrator import GridSample


def get_vtk_vector_field(image):
    vertices = template.verts_packed().unsqueeze(0)
    grid_sampler = GridSample("bilinear")

    image = model._fix_input_shape(image)
    assert image.shape[0] == 1

    with torch.no_grad():
        pre_encoding = model.get_encoder_input(image)
        lt_deformed_vertices = model.pretrained_linear_transform(image, vertices)
        encoding = model.encoder(pre_encoding)
        encoding = model.get_flow_decoder_input(encoding, lt_deformed_vertices, 1, model.input_size)
        flow = model.flow_decoder(encoding)

    interpolated_flow = grid_sampler.interpolate(flow, lt_deformed_vertices)
    lt_deformed_vertices = lt_deformed_vertices.squeeze(0)
    deformed_template = template.update_packed(lt_deformed_vertices)

    np_vector_field = interpolated_flow.numpy()
    vtk_data_array = numpy_to_vtk(num_array=np_vector_field.ravel())
    vtk_data_array.SetNumberOfComponents(3)
    vtk_data_array.SetName("flow-field")

    vtk_template = deformed_template.to_vtk_mesh()
    vtk_template.GetPointData().AddArray(vtk_data_array)

    return vtk_template


def get_sample(sample_name):
    sample_path = os.path.join(dataset_folder, sample_name + ".pkl")
    assert os.path.isfile(sample_path)
    sample = pickle.load(open(sample_path, "rb"))
    return sample


root_dir = "output/WholeHeartData/trained_models/mr/flow/model-1/"
dataset_folder = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/mr/processed/indexed/data"
template_file = "data/template/highres_template.vtp"

output_root_dir = os.path.join(root_dir, "evaluation", "meshes", "incremental")
model_file = os.path.join(root_dir, "best_model.pth")
sample_name = "sample03"

data = torch.load(model_file, map_location=torch.device("cpu"))
model = data["model"]

template = Template.from_vtk(template_file)


sample = get_sample(sample_name)
image = sample["image"]
vtk_template = get_vtk_vector_field(image)


output_file = os.path.join(output_root_dir, sample_name + "_flow.vtp")
vtu.write_vtk_polydata(vtk_template, output_file)
