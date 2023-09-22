import pytorch3d.structures
import torch
import os
import sys
import pytorch3d
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.loss.point_mesh_distance import point_face_distance, _DEFAULT_MIN_TRIANGLE_AREA
from pytorch3d.structures import Pointclouds, Meshes
import SimpleITK as sitk

sys.path.append(os.getcwd())
from src.template import Template


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def make_3d_grid(num_voxels, start=-1.0, stop=1.0):
    """Generate a 3D grid of points corresponding to normalized voxel coordinates

    Args:
        num_voxels: int
            Number of voxels per dimension in the image (image assumed to be a cube)
        start: int | optional
            lower bound of normalized coordinates. Default = -1 consistent with pytorch grid_sample
        stop: int | optional
            upper bound of normalized coordinates. Default = +1 consistent with pytorch grid_sample

    Returns: torch.Tensor

    """
    voxel_size = (stop - start) / num_voxels
    start_coord = start + voxel_size / 2
    stop_coord = stop - voxel_size / 2

    xrange = torch.linspace(start_coord, stop_coord, num_voxels)
    z, y, x = torch.meshgrid([xrange, xrange, xrange], indexing="ij")
    meshgrid = torch.stack([x, y, z])
    meshgrid = meshgrid.permute([1, 2, 3, 0])
    return meshgrid


def distance_to_face(
        meshes: Meshes,
        pcls: Pointclouds,
        min_triangle_area: float = _DEFAULT_MIN_TRIANGLE_AREA,
):
    print("Computing distance:")

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area
    )

    return point_to_face.sqrt()


if __name__ == "__main__":
    mesh_fn = "data/template/highres_template.vtp"
    grid_size = 128

    template = Template.from_vtk(mesh_fn).to(device)
    mesh = pytorch3d.structures.join_meshes_as_scene(template)

    grid = make_3d_grid(grid_size, start=0, stop=1)
    grid = grid.reshape([-1, 3]).to(device)

    point_cloud = Pointclouds([grid]).to(device)

    dists = distance_to_face(mesh, point_cloud)
    dists = dists.reshape(3 * [grid_size]).cpu().numpy()

    img = sitk.GetImageFromArray(dists)
    img.SetSpacing(3 * [1 / grid_size])
    sitk.WriteImage(img, "test/output/distance.vtk")
