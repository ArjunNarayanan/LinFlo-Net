import vtk_utils.vtk_utils as vtu
from src.template import Template
import src.io_utils as io
import torch
from torch.nn.functional import one_hot

template_fn = "../data/template/highres_template.vtp"
image_fn = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/ct/processed/vtk_image/CT_1.vti"

ref_image = vtu.load_vtk_image(image_fn)
template_mesh = vtu.load_vtk_mesh(template_fn)

seg = vtu.multiclass_convert_polydata_to_imagedata(template_mesh, ref_image)
seg_arr = io.vtk_image_to_torch(seg).to(torch.long)

seg_arr_one_hot = one_hot(seg_arr)
seg_arr = seg_arr_one_hot.permute([3, 0, 1, 2])
seg_arr = seg_arr[1:]

output_file = "../data/highres_template_segmentation.pth"
torch.save(seg_arr, output_file)

# outfile = "output/template_segmentation.vti"
# vtu.write_vtk_image(seg, outfile)
