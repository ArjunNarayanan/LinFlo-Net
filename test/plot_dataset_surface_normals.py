import os
import pickle
from pytorch3d.structures import Meshes
from src.io_utils import pytorch3d_to_vtk
from src.template import Template
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk_utils.vtk_utils as vtu


def get_sample(sample_name):
    sample_path = os.path.join(dataset_folder, sample_name + ".pkl")
    assert os.path.isfile(sample_path)
    sample = pickle.load(open(sample_path, "rb"))
    return sample


dataset_folder = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/mr/processed/indexed/data"
sample_name = "sample03"

sample = get_sample(sample_name)
meshes = sample["meshes"]

verts = [m.verts_packed() for m in meshes]
faces = [m.faces_packed() for m in meshes]

mesh = Template(verts, faces)
normals = mesh.verts_normals_packed()

np_normals = normals.numpy()
vtk_data_array = numpy_to_vtk(np_normals.ravel())
vtk_data_array.SetNumberOfComponents(3)
vtk_data_array.SetName("normals")

vtk_mesh = mesh.to_vtk_mesh()
vtk_mesh.GetPointData().AddArray(vtk_data_array)

output_file = os.path.join("test/output", sample_name + ".vtp")
vtu.write_vtk_polydata(vtk_mesh, output_file)