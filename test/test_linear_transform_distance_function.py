import torch
import src.io_utils as io
from src.linear_transform import linear_transform_image, LinearTransformer
from src.template import Template
from src.integrator import GridSample
from monai.transforms import LoadImage
import SimpleITK as sitk
import vtk_utils.vtk_utils as vtu

translations = torch.tensor([[0.1, 0.2, -0.3]])
scale = torch.tensor([[0.8, 1.2, 1.3]])
rotate = torch.tensor([[0.2, -0.3, 0.1]])


template_distance_file = "data/template/highres_template_distance.vtk"

# vtk_distance_map = io.vtu.load_vtk_image(template_distance_file)
# torch_distance_map = io.vtk_image_to_torch(vtk_distance_map)

# monai_loader = LoadImage(image_only=True)
# monai_distance_map = monai_loader(template_distance_file)
# monai_distance_map = monai_distance_map.unsqueeze(0).unsqueeze(0)

torch_distance_map = io.read_image(template_distance_file)
torch_distance_map = torch_distance_map.unsqueeze(0).unsqueeze(0)

template_file = "data/template/highres_template.vtp"
template = Template.from_vtk(template_file)

interpolater = GridSample("bilinear")

original_vertices = template.verts_packed().unsqueeze(0)
original_verts_distances = interpolater.interpolate(torch_distance_map, original_vertices)

torch_distance_map = torch_distance_map.squeeze(0)
transformed_distance_map = linear_transform_image(
    torch_distance_map,
    scale.squeeze(0),
    translations.squeeze(0),
    rotate.squeeze(0)
)


mesh_linear_transform = LinearTransformer(scale, translations, rotate)
mesh_vertices = template.verts_packed()
transformed_vertices = mesh_linear_transform.transform(mesh_vertices)
transformed_template = template.update_packed(transformed_vertices)

transformed_vertices = transformed_vertices.unsqueeze(0)
transformed_distance_map = transformed_distance_map.unsqueeze(0)
transformed_verts_distances = interpolater.interpolate(transformed_distance_map, transformed_vertices)

original_at_transformed = interpolater.interpolate(transformed_distance_map, original_vertices)

original_at_transformed_max = original_at_transformed.max()
original_max = original_verts_distances.abs().max()
transformed_max = transformed_verts_distances.abs().max()

print("Original max : ", original_max)
print("Transformed max : ", transformed_max)
print("Original at transformed : ", original_at_transformed_max)

transformed_distance_map = transformed_distance_map.squeeze(0).squeeze(0).cpu().numpy()
transformed_distance_map = transformed_distance_map.transpose(2, 1, 0)
sitk_distance_map = sitk.GetImageFromArray(transformed_distance_map)
grid_size = torch_distance_map.shape[-1]
sitk_distance_map.SetSpacing(3 * [1 / grid_size])
out_img_fn = "test/output/translated_distance.vtk"
sitk.WriteImage(sitk_distance_map, out_img_fn)


# io.write_image(transformed_distance_map, vtk_distance_map, out_img_fn)


vtu.write_vtk_polydata(transformed_template.to_vtk_mesh(), "test/output/translated_template.vtp")
