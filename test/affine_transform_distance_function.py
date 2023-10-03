import SimpleITK as sitk
import os
import sys
import torch

sys.path.append(os.getcwd())
from src.template import Template
from src.linear_transform import LinearTransformer, linear_transform_image
import vtk_utils.vtk_utils as vtu
from src.dataset import ImageSegmentationMeshDataset
import src.io_utils as io
from src.integrator import GridSample

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def get_linear_transform_parameters(image):
    with torch.no_grad():
        parameters = model.linear_transform_parameters_from_image(image)

    # parameters = (p.squeeze(0) for p in parameters)
    return parameters


def print_max_distance_value(vertices, distance_map):
    vertices = vertices.unsqueeze(0)
    distance_map = distance_map.unsqueeze(0)
    sampler = GridSample("bilinear")
    interpolated_distances = sampler.interpolate(distance_map, vertices)
    print("MIN INTERP DISTANCE : ", interpolated_distances.min().item())
    print("MAX INTERP DISTANCE : ", interpolated_distances.max().item())
    print("AVG INTERP DISTANCE : ", interpolated_distances.mean().item())


def transform_mesh_and_distance_map(image):
    template_coords = template.verts_packed().unsqueeze(0).to(device)

    scale, translate, rotate = get_linear_transform_parameters(image)
    mesh_transformer = LinearTransformer(scale, translate, rotate)
    deformed_coords = mesh_transformer.transform(template_coords)

    deformed_coords = deformed_coords.squeeze(0).cpu()
    deformed_mesh = template.update_packed(deformed_coords).to_vtk_mesh()

    scale, translate, rotate = (p.squeeze(0) for p in (scale, translate, rotate))
    transformed_distance_map = linear_transform_image(
        distance_map,
        scale,
        translate,
        rotate,
        interpolation
    )

    # print_max_distance_value(deformed_coords, transformed_distance_map)

    transformed_distance_map = transformed_distance_map.squeeze(0).cpu().numpy()
    transformed_distance_map = transformed_distance_map.transpose(2, 1, 0)
    sitk_distance_map = sitk.GetImageFromArray(transformed_distance_map)
    grid_size = image.shape[-1]
    sitk_distance_map.SetSpacing(3 * [1 / grid_size])

    return deformed_mesh, sitk_distance_map


def process_file(file_index):
    filename = dataset.get_file_name(file_index)
    print("Processing file : ", filename)

    sample = dataset[file_index]
    image = sample["image"].to(device)

    deformed_mesh, transformed_map = transform_mesh_and_distance_map(image)

    output_filename = "sample-" + str(file_index)
    output_mesh_file = os.path.join(output_mesh_dir, output_filename + ".vtp")
    vtu.write_vtk_polydata(deformed_mesh, output_mesh_file)
    output_distance_file = os.path.join(output_distance_map_dir, output_filename + ".vtk")
    sitk.WriteImage(transformed_map, output_distance_file)
    print("\n")
    # vtu.write_vtk_image(transformed_map, output_distance_file)


if __name__ == "__main__":
    model_fn = "../output/WholeHeartData/ct/linear-transform/augment-20/best_model.pth"
    data_dir = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/ct/processed"
    output_dir = "../output/WholeHeartData/ct/linear-transform/augment-20/transformed-segmentation-maps"
    interpolation = "nearest"

    # template_distance_file = "../data/template/highres_template_distance.vtk"
    template_distance_file = "../data/template/highres_template_segmentation.vti"
    template_file = "../data/template/highres_template.vtp"

    template = Template.from_vtk(template_file)
    # distance_map = sitk.ReadImage(template_distance_file)
    # distance_map = torch.tensor(sitk.GetArrayFromImage(distance_map)).unsqueeze(0).to(device)
    distance_map = io.read_image(template_distance_file).unsqueeze(0).to(device)

    dataset = ImageSegmentationMeshDataset(data_dir)

    model = torch.load(model_fn, map_location="cpu")["model"]
    model.eval()
    model.to(device)

    output_mesh_dir = os.path.join(output_dir, "meshes")
    output_distance_map_dir = os.path.join(output_dir, "distances")
    if not os.path.isdir(output_mesh_dir):
        os.makedirs(output_mesh_dir)
    if not os.path.isdir(output_distance_map_dir):
        os.makedirs(output_distance_map_dir)

    for file_index in range(len(dataset)):
        process_file(file_index)
