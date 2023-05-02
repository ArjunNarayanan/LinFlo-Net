from src.template import *
import vtk_utils.vtk_utils as vtu

fn = "data/template/highres_template.vtp"
template = Template.from_vtk(fn)

lv = template[0]
lv_vtk = lv.to_vtk_mesh()
vtu.write_vtk_polydata(lv_vtk, "data/template/lv_template.vtp")