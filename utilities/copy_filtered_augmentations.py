import os
import sys
import shutil

def clear_and_make_directory(dir):
    return

input_dir = "/global/scratch/users/arjunnarayanan/WholeHeartData/train/ct/augment20/processed/"
output_dir = "/global/scratch/users/arjunnarayanan/WholeHeart-ct-crop-mr/train"

input_img_dir = os.path.join(input_dir, "vtk_image")
input_seg_dir = os.path.join(input_dir, "vtk_segmentation")
input_mesh_dir = os.path.join(input_dir, "vtk_mesh")

assert os.path.isdir(input_img_dir)
assert os.path.isdir(input_seg_dir)
assert os.path.isdir(input_mesh_dir)

output_img_dir = os.path.join(output_dir, "vtk_image")
output_seg_dir = os.path.join(output_dir, "vtk_segmentation")
output_mesh_dir = os.path.join(output_dir, "vtk_mesh")
