import os
import sys
import glob

sys.path.append(os.getcwd())
import vtk_utils.vtk_utils as vtu
import SimpleITK as sitk
import numpy as np
import time


def generate_mesh(vtk_seg, seg_ids, decimation_ratio=0.8):
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


input_seg = "/Users/arjun/Documents/Research/SimCardio/Datasets/WholeHeartData/validation/ct/label/CT_1.nii.gz"

start = time.perf_counter()
seg = sitk.ReadImage(input_seg)
stop = time.perf_counter()
print("Elapsed : ", stop - start)

seg_arr = sitk.GetArrayFromImage(seg)
labels = np.unique(seg_arr)

start = time.perf_counter()
vtk_seg = vtu.exportSitk2VTK(seg)[0]
stop = time.perf_counter()
print("Elapsed time : ", stop - start)

start = time.perf_counter()
mesh = generate_mesh(vtk_seg, labels)
stop = time.perf_counter()
print("Elapsed : ", stop - start)