from src.template import Template, BatchTemplate
from src.dataset import image_segmentation_mesh_dataloader
import torch
import vtk_utils.vtk_utils as vtu
from src.linear_transform import LinearTransformer
import os


def get_random_scaling(batch_size, magnitude):
    perturbation = magnitude * (2 * torch.rand([batch_size, 3]) - 1)
    perturbation = 1 - perturbation
    return perturbation


def get_random_translation(batch_size, magnitude):
    perturbation = magnitude * (2 * torch.rand([batch_size, 3]) - 1)
    return perturbation


def get_random_rotation(batch_size, magnitude):
    perturbation = 2 * torch.pi * magnitude * (2 * torch.rand([batch_size, 3]) - 1)
    return perturbation


def get_random_linear_transformer(batch_size, scale_magnitude, translate_magnitude, rotate_magnitude):
    scale_params = get_random_scaling(batch_size, scale_magnitude)
    translate_params = get_random_translation(batch_size, translate_magnitude)
    rotate_params = get_random_rotation(batch_size, rotate_magnitude)
    return LinearTransformer(scale_params, translate_params, rotate_params)


scale_perturb = 0.2
translate_perturb = 0.1
rotate_perturb = 0.1
batch_size = 5

model_fn = "output/linear_transform/best_model_dict.pth"
model_data = torch.load(model_fn, map_location=torch.device("cpu"))
model = model_data["model"]

dataset_fn = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartDataSegmentation/validation"
dataset = image_segmentation_mesh_dataloader(dataset_fn, batch_size=batch_size)

template_fn = "data/template/highres_template.vtp"
template = Template.from_vtk(template_fn)
batched_template = BatchTemplate.from_single_template(template, batch_size)
batched_verts = batched_template.batch_vertex_coordinates()

data = next(iter(dataset))
image = data["image"]
deformed_verts = model(image, batched_verts)

perturbation = get_random_linear_transformer(batch_size, scale_perturb, translate_perturb, rotate_perturb)

perturbed_verts = [perturbation.transform(v) for v in deformed_verts]
batched_template.update_batched_vertices(perturbed_verts)

component_meshes = batched_template.join_component_meshes()
component_meshes_as_template = [Template(m.verts_padded(), m.faces_padded()) for m in component_meshes]
vtk_meshes = [t.to_vtk_mesh() for t in component_meshes_as_template]

output_folder = "output/test"
for (idx, mesh) in enumerate(vtk_meshes):
    filename = os.path.join(output_folder, "test-" + str(idx) + ".vtp")
    vtu.write_vtk_polydata(mesh, filename)