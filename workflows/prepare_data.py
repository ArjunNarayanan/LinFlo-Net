import os
import glob

from src.pre_process import resample_spacing, rescale_intensity
import vtk_utils.vtk_utils as vtu
import SimpleITK as sitk
import argparse
import numpy as np
import yaml


def process_im_and_seg(im_fn, seg_fn, size, modality):
    img = sitk.ReadImage(im_fn)
    seg = sitk.ReadImage(seg_fn)
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr[seg_arr == 421] = 420
    seg_new = sitk.GetImageFromArray(seg_arr)
    seg_new.CopyInformation(seg)
    # make sure img and seg match
    seg = sitk.Resample(seg_new, img.GetSize(),
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        img.GetOrigin(),
                        img.GetSpacing(),
                        img.GetDirection(),
                        0,
                        seg.GetPixelID())
    new_img = resample_spacing(img, template_size=[size] * 3, order=1)[0]
    spacing = 1. / float(size)
    new_img.SetSpacing([spacing] * 3)
    vtk_img = vtu.exportSitk2VTK(new_img)[0]
    py_arr = vtu.vtk_to_numpy(vtk_img.GetPointData().GetScalars())
    py_arr = rescale_intensity(py_arr, modality, [750, -750])
    vtk_img.GetPointData().SetScalars(vtu.numpy_to_vtk(py_arr))

    new_seg = resample_spacing(seg, template_size=[256] * 3, order=0)[0]
    spacing = 1. / float(256)
    new_seg.SetSpacing([spacing] * 3)
    vtk_seg = vtu.exportSitk2VTK(new_seg)[0]
    return vtk_img, vtk_seg


def pre_process_lv(im_fn, seg_fn, size, modality):
    vtk_img, vtk_seg = process_im_and_seg(im_fn, seg_fn, size, modality)
    mesh = vtu.vtk_marching_cube(vtk_seg, bg_id=0, seg_id=3)
    mesh = vtu.smooth_polydata(vtu.decimation(mesh, 0.2), 50)
    mesh = vtu.smooth_polydata(vtu.decimation(mesh, 0.2), 50)
    mesh = vtu.get_all_connected_polydata(mesh)
    r_id_array = [0] * mesh.GetNumberOfPoints()
    r_id_array = vtu.numpy_to_vtk(r_id_array)
    r_id_array.SetName('RegionId')
    mesh.GetPointData().AddArray(r_id_array)

    return vtk_img, vtk_seg, mesh


def pre_process_wh(im_fn, seg_fn, size, modality):
    vtk_img, vtk_seg = process_im_and_seg(im_fn, seg_fn, size, modality)
    seg_ids = np.unique(vtu.vtk_to_numpy(vtk_seg.GetPointData().GetScalars()))
    bg_id = 0
    mesh_list = []
    r_id_array = []
    for s_id in seg_ids:
        if s_id != bg_id:
            mesh = vtu.vtk_marching_cube(vtk_seg, bg_id=bg_id, seg_id=s_id)
            mesh = vtu.fill_hole(mesh)
            mesh = vtu.smooth_polydata(mesh, 30, smoothingFactor=0.5)
            mesh = vtu.smooth_polydata(vtu.decimation(mesh, 0.8), 50)
            mesh = vtu.smooth_polydata(mesh, 30, smoothingFactor=0.5)
            r_id_array += [s_id] * mesh.GetNumberOfPoints()
            mesh_list.append(mesh)
    mesh_all = vtu.appendPolyData(mesh_list)
    r_id_array = vtu.numpy_to_vtk(r_id_array)
    r_id_array.SetName('RegionId')
    mesh_all.GetPointData().AddArray(r_id_array)
    return vtk_img, vtk_seg, mesh_all


def check_image_mesh_alignment(img, mesh):
    x, y, z = img.GetDimensions()
    py_img = vtu.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0)
    coords = vtu.vtk_to_numpy(mesh.GetPoints().GetData())
    coords *= np.expand_dims(np.array(py_img.shape), 0)
    coords = coords.astype(int)
    py_img[coords[:, 0], coords[:, 1], coords[:, 2]] = 1000.
    py_img = py_img.transpose(2, 1, 0).reshape(z * y * z)
    img.GetPointData().SetScalars(vtu.numpy_to_vtk(py_img))
    return img


def process_and_write_output(im_folder, seg_folder, out_folder, in_fn, modality, size=128):
    im_fn = os.path.join(im_folder, in_fn)
    seg_fn = os.path.join(seg_folder, in_fn)

    print("Processing file : ", im_fn)

    new_img, new_seg, mesh = pre_process_wh(im_fn, seg_fn, size, modality)

    fn_no_ext = in_fn.split(".")[0]
    vtu.write_vtk_image(new_img, os.path.join(out_folder, "vtk_image", fn_no_ext + ".vti"))
    vtu.write_vtk_image(new_seg, os.path.join(out_folder, "seg", fn_no_ext + ".vti"))
    vtu.write_vtk_polydata(mesh, os.path.join(out_folder, "vtk_mesh", fn_no_ext + ".vtp"))


def process_folder(im_folder, seg_folder, out_folder, modality, extension):
    out_im_folder = os.path.join(out_folder, "vtk_image")
    out_seg_folder = os.path.join(out_folder, "seg")
    out_mesh_folder = os.path.join(out_folder, "vtk_mesh")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(out_im_folder):
        os.makedirs(out_im_folder)
    if not os.path.exists(out_seg_folder):
        os.makedirs(out_seg_folder)
    if not os.path.exists(out_mesh_folder):
        os.makedirs(out_mesh_folder)

    filenames = glob.glob(os.path.join(im_folder, "*" + extension))
    print("Detected ", len(filenames), " files in image folder")

    for file_path in filenames:
        file_name = os.path.basename(file_path)
        assert os.path.isfile(os.path.join(seg_folder, file_name))
        process_and_write_output(im_folder, seg_folder, out_folder, file_name, modality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare Images and Meshes")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)

    im_folder = config["im_folder"]
    seg_folder = config["seg_folder"]
    out_folder = config["out_folder"]
    modality = config["modality"]
    extension = config["extension"]

    process_folder(im_folder, seg_folder, out_folder, modality, extension)
