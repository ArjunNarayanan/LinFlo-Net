from src.template import Template
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk_utils.vtk_utils as vtu


template_file = "data/template/highres_template.vtp"
template = Template.from_vtk(template_file)

normals = template.verts_normals_packed()
np_normals = normals.numpy()

vtk_data_array = numpy_to_vtk(np_normals.ravel())
vtk_data_array.SetNumberOfComponents(3)
vtk_data_array.SetName("normals")

vtk_template = template.to_vtk_mesh()
vtk_template.GetPointData().AddArray(vtk_data_array)

output_file = "test/output/template_with_normals.vtp"
vtu.write_vtk_polydata(vtk_template, output_file)

