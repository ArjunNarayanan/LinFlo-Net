from src.template import Template
from src.io_utils import *
import vtk_utils.vtk_utils as vtu

wh_filename = "data/template/highres_template.vtp"
ao_filename = "data/template/aortic_root.vtp"

wh_meshes = read_mesh_list(wh_filename)
ao_meshes = read_mesh_list(ao_filename)

mesh_list = wh_meshes + ao_meshes
verts = [mesh2verts(m) for m in mesh_list]
faces = [mesh2faces(m) for m in mesh_list]

template = Template(verts, faces)
vtk_template = template.to_vtk_mesh()

vtu.write_vtk_polydata(vtk_template, "data/template/whole_heart_with_ao.vtp")