import os
import sys
import glob
sys.path.append(os.getcwd())
import vtk_utils.vtk_utils as vtu
import SimpleITK as sitk
import numpy as np
import argparse

def generate_mesh(vtk_seg, decimation_ratio=0.8):
    seg_ids = np.unique(vtu.vtk_to_numpy(vtk_seg.GetPointData().GetScalars()))
    bg_id = 0
    mesh_list = []
    r_id_array = []
    for s_id in seg_ids:
        if s_id != bg_id:
            mesh = vtu.vtk_marching_cube(vtk_seg, bg_id=bg_id, seg_id=s_id)
            mesh = vtu.fill_hole(mesh)
            mesh = vtu.smooth_polydata(mesh, 30, smoothingFactor=0.5)
            mesh = vtu.smooth_polydata(vtu.decimation(mesh, decimation_ratio), 50)
            mesh = vtu.smooth_polydata(mesh, 30, smoothingFactor=0.5)
            r_id_array += [s_id] * mesh.GetNumberOfPoints()
            mesh_list.append(mesh)
    mesh_all = vtu.appendPolyData(mesh_list)
    r_id_array = vtu.numpy_to_vtk(r_id_array)
    r_id_array.SetName('RegionId')
    mesh_all.GetPointData().AddArray(r_id_array)
    return mesh_all

def generate_mesh_from_seg(seg_fn):
    seg = sitk.ReadImage(seg_fn)
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr[seg_arr == 421] = 420
    seg_new = sitk.GetImageFromArray(seg_arr)
    seg_new.CopyInformation(seg)

    vtk_seg = vtu.exportSitk2VTK(seg_new)[0]
    mesh = generate_mesh(vtk_seg)
    return mesh

def write_all_meshes(filenames, output_dir):
    for seg_fn in filenames:
        mesh = generate_mesh_from_seg(seg_fn)
        fn = os.path.basename(seg_fn).split(".")[0]
        out_fn = os.path.join(output_dir, fn + ".vtp")

        print("Writing mesh file : ", out_fn)
        vtu.write_vtk_polydata(mesh, out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth validation meshes via marching cube")
    parser.add_argument("-f", help="Input folder with ground-truth segmentations")
    parser.add_argument("-o", help="Output folder")
    parser.add_argument("-e", help="Input file extension")
    args = parser.parse_args()

    input_folder = args.f 
    output_folder = args.o
    extension = args.e

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    
    filenames = glob.glob(os.path.join(input_folder, "*" + extension))
    write_all_meshes(filenames, output_folder)