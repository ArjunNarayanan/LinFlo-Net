import os
import sys
import pandas as pd
import torch
import numpy as np
import vtk
import yaml
from pytorch3d.structures import Meshes

sys.path.append(os.getcwd())
import vtk_utils.vtk_utils as vtu


class SaveBestModel:
    def __init__(self, root_dir, best_valid_loss=float("inf")):
        self.root_dir = root_dir
        self.best_validation_loss = best_valid_loss
        self.best_model_file = os.path.join(self.root_dir, "best_model.pth")

    def save_best_model(self, current_validation_loss, epoch, data):
        if current_validation_loss < self.best_validation_loss:
            self.best_validation_loss = current_validation_loss
            print(f"\nBest validation loss: {self.best_validation_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save(data, self.best_model_file)

    def save_checkpoint(self, epoch, data):
        print(f"\nSaving checkpoint at epoch {epoch}")
        checkpoint_file = os.path.join(self.root_dir, "model-" + str(epoch) + ".pth")
        torch.save(data, checkpoint_file)

    def save_loss(self, loss_dict, filename):
        output_file = os.path.join(self.root_dir, filename)
        df = pd.DataFrame(loss_dict)
        df.to_csv(output_file, index=False)


def vtk_image_to_torch(img):
    x, y, z = img.GetDimensions()
    py_img = vtu.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
    torch_img = torch.tensor(py_img)
    return torch_img


def read_image(img_fn):
    img = vtu.load_vtk_image(img_fn)
    torch_img = vtk_image_to_torch(img)
    return torch_img


def torch_to_vtk_image(torch_img, ref_img):
    py_img = torch_img.detach().numpy()
    out_img = py_img.transpose(2, 1, 0).reshape(-1)
    ref_img.GetPointData().SetScalars(vtu.numpy_to_vtk(out_img))
    return ref_img


def write_image(py_img, ref_img, outfile):
    out_img = py_img.transpose(2, 1, 0).reshape(-1)
    ref_img.GetPointData().SetScalars(vtu.numpy_to_vtk(out_img))
    vtu.write_vtk_image(ref_img, outfile)


def read_mesh_list(mesh_fn):
    mesh = vtu.load_vtk_mesh(mesh_fn)
    rids = mesh.GetPointData().GetArray("RegionId")
    if rids is None:
        return [mesh]
    else:
        region_ids = np.unique(rids)
        mesh_list = []
        for i in region_ids:
            mesh_i = vtu.thresholdPolyData(mesh, 'RegionId', (i, i), 'point')
            mesh_list.append(mesh_i)
        return mesh_list


def mesh2verts(mesh):
    verts = torch.tensor(vtu.vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32))
    return verts


def mesh2faces(mesh):
    faces = vtu.vtk_to_numpy(mesh.GetPolys().GetData())
    faces = faces.reshape([-1, 4])
    faces = faces[:, 1:]
    return torch.tensor(faces)


def pytorch3d_meshes_from_vtk(mesh_fn, device=None):
    mesh_list = read_mesh_list(mesh_fn)
    verts = [mesh2verts(m).unsqueeze(0) for m in mesh_list]
    faces = [mesh2faces(m).unsqueeze(0) for m in mesh_list]

    meshes = [Meshes(verts=v, faces=f) for (v, f) in zip(verts, faces)]
    if device is not None:
        meshes = [m.to(device) for m in meshes]

    return meshes


def read_face_ids(mesh_fn, id_name):
    mesh = vtu.load_vtk_mesh(mesh_fn)
    fids = mesh.GetCellData().GetArray(id_name)
    assert fids is not None, "Did not find cell data with name " + id_name
    fids = vtu.vtk_to_numpy(fids)
    return fids


def construct_region_ids_array(num_verts_per_mesh):
    rids = np.zeros(num_verts_per_mesh.sum(), dtype=int)
    start = 0
    stop = 0
    for (idx, nv) in enumerate(num_verts_per_mesh):
        stop = stop + nv
        rids[start:stop] = idx + 1
        start = stop
    return rids


def create_vtk_points(np_points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtu.numpy_to_vtk(np_points))
    return vtk_points


def create_face(face_connectivity):
    tri = vtk.vtkTriangle()
    ids = tri.GetPointIds()
    for i in range(3):
        ids.SetId(i, face_connectivity[i])
    return tri


def create_vtk_faces(connectivity):
    faces = vtk.vtkCellArray()
    for i in range(connectivity.shape[0]):
        tri = create_face(connectivity[i])
        faces.InsertNextCell(tri)
    return faces


def create_vtk_mesh(np_points, np_faces, rids_array=None, faceids=None, faceids_name=None):
    vtk_points = create_vtk_points(np_points)
    vtk_faces = create_vtk_faces(np_faces)

    mesh = vtk.vtkPolyData()
    mesh.SetPoints(vtk_points)
    mesh.SetPolys(vtk_faces)

    if rids_array is not None:
        assert len(rids_array) == np_points.shape[0]
        vtk_rids = vtu.numpy_to_vtk(rids_array)
        vtk_rids.SetName("RegionId")
        mesh.GetPointData().AddArray(vtk_rids)

    if faceids is not None:
        assert len(faceids) == np_faces.shape[0]
        assert faceids_name is not None
        vtk_fids = vtu.numpy_to_vtk(faceids)
        vtk_fids.SetName(faceids_name)
        mesh.GetCellData().AddArray(vtk_fids)

    return mesh


def pytorch3d_to_vtk(pytorch3d_mesh):
    np_points = pytorch3d_mesh.verts_packed().cpu().detach().numpy()
    np_faces = pytorch3d_mesh.faces_packed().cpu().detach().numpy()
    np_rids = construct_region_ids_array(pytorch3d_mesh.num_verts_per_mesh())
    faceids = pytorch3d_mesh.faceids
    faceids_name = pytorch3d_mesh.faceids_name
    mesh = create_vtk_mesh(np_points, np_faces, rids_array=np_rids, faceids=faceids, faceids_name=faceids_name)
    return mesh


def loss2str(loss_components):
    out_str = ""
    if "chamfer_distance" in loss_components:
        out_str += "CHD {:1.3e} | ".format(loss_components["chamfer_distance"])
    if "chamfer_normal" in loss_components:
        out_str += "CHN {:1.3e} | ".format(loss_components["chamfer_normal"])
    if "divergence" in loss_components:
        out_str += "DIV {:1.3e} | ".format(loss_components["divergence"])
    if "cross_entropy" in loss_components:
        out_str += "MCE {:1.3e} | ".format(loss_components["cross_entropy"])
    if "dice" in loss_components:
        out_str += "DIC {:1.3e} | ".format(loss_components["dice"])
    if "edge" in loss_components:
        out_str += "EDG {:1.3e} | ".format(loss_components["edge"])
    if "laplace" in loss_components:
        out_str += "LAP {:1.3e} | ".format(loss_components["laplace"])
    if "normal" in loss_components:
        out_str += "NOR {:1.3e} | ".format(loss_components["normal"])

    out_str += " TOT {:1.3e} | ".format(loss_components["total"])

    return out_str


def load_yaml_config(config_fn):
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
