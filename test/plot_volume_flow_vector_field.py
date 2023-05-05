import torch
from src.template import Template
import os
import pickle
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def vector_field_to_vtk(np_vector_field, output_file):
    assert np_vector_field.ndim == 4
    assert np_vector_field.shape[-1] == 3
    assert all(s == np_vector_field.shape[0] for s in np_vector_field.shape[:-1])

    grid_size = np_vector_field.shape[0]
    assert grid_size > 1
    spacing = 3 * [1.0 / (grid_size - 1)]

    image = vtk.vtkImageData()
    image.SetOrigin(0., 0., 0.)
    image.SetSpacing(spacing)
    image.SetDimensions(grid_size, grid_size, grid_size)

    vtk_data_array = numpy_to_vtk(num_array=np_vector_field.ravel())
    vtk_data_array.SetNumberOfComponents(3)
    vtk_data_array.SetName("vector-field")
    image.GetPointData().SetScalars(vtk_data_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetInputData(image)
    writer.SetFileName(output_file)
    print("Writing vector field to : ", output_file)
    writer.Write()


def get_sample(sample_name):
    sample_path = os.path.join(dataset_folder, sample_name + ".pkl")
    assert os.path.isfile(sample_path)
    sample = pickle.load(open(sample_path, "rb"))
    return sample


def get_flow_vector_field(image, vertices):
    image = model._fix_input_shape(image)
    assert image.shape[0] == 1

    with torch.no_grad():
        pre_encoding = model.get_encoder_input(image)
        lt_deformed_vertices = model.pretrained_linear_transform(image, vertices)
        encoding = model.encoder(pre_encoding)
        encoding = model.get_flow_decoder_input(encoding, lt_deformed_vertices, 1, model.input_size)
        flow = model.flow_decoder(encoding)

    flow = flow.squeeze(0).detach().numpy()
    return flow



# root_dir = "output/WholeHeartData/trained_models/mr/flow/model-1/"
# dataset_folder = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/mr/processed/indexed/data"
# template_file = "data/template/highres_template.vtp"
#
# output_root_dir = os.path.join(root_dir, "evaluation", "meshes", "incremental")
# model_file = os.path.join(root_dir, "best_model.pth")
# sample_name = "sample03"
#
# data = torch.load(model_file, map_location=torch.device("cpu"))
# model = data["model"]
#
# template = Template.from_vtk(template_file)
# template_vertices = template.verts_packed().unsqueeze(0)
#
# sample = get_sample(sample_name)
# image = sample["image"]
#
# flow = get_flow_vector_field(image, template_vertices)

out_flow = np.copy(flow)
# out_flow = np.flip(out_flow, axis=0)
out_flow = out_flow.transpose([3,2,1,0])

output_file = os.path.join(output_root_dir, sample_name + "_flow.vti")
vector_field_to_vtk(out_flow, output_file)