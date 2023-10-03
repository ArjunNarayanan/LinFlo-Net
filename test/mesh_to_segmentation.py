import vtk_utils.vtk_utils as vtu
from src.io_utils import read_image
from src.template import Template

template_fn = "../data/template/highres_template.vtp"
image_fn = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/ct/processed/vtk_image/CT_1.vti"

ref_image = vtu.load_vtk_image(image_fn)
template_mesh = vtu.load_vtk_mesh(template_fn)

seg = vtu.multiclass_convert_polydata_to_imagedata(template_mesh, ref_image)

outfile = "output/template_segmentation.vti"
vtu.write_vtk_image(seg, outfile)