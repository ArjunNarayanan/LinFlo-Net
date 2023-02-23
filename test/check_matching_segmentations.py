import SimpleITK as sitk
import src.pre_process as pre
import vtk_utils.vtk_utils as vtu

size = 128
modality = "ct"
img_fn1 = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartData/raw_validation/ct/image/CT_1.nii.gz"
img = sitk.ReadImage(img_fn1)
new_img = pre.resample_spacing(img, 3 * [size], order=1)[0]
spacing = 1. / float(size)
new_img.SetSpacing([spacing] * 3)
vtk_img = vtu.exportSitk2VTK(new_img)[0]

py_arr = vtu.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
py_arr = pre.rescale_intensity(py_arr, modality, [750, -750])
vtk_img.GetPointData().SetScalars(vtu.numpy_to_vtk(py_arr))
