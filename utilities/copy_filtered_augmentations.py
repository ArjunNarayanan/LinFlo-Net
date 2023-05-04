import os
import sys
import shutil
import glob


def make_output_directory(dir):
    if not os.path.isdir(dir):
        print("Creating output directory : ", dir)
        os.makedirs(dir)


def move_sample(sample_name):
    sample_img = os.path.join(input_img_dir, sample_name + ".vti")
    sample_seg = os.path.join(input_seg_dir, sample_name + ".vti")
    sample_mesh = os.path.join(input_mesh_dir, sample_name + ".vtp")
    sample_data = os.path.join(input_data_dir, sample_name + ".pkl")

    assert os.path.isfile(sample_img)
    assert os.path.isfile(sample_seg)
    assert os.path.isfile(sample_mesh)
    assert os.path.isfile(sample_data)

    output_img = os.path.join(output_img_dir, sample_name + ".vti")
    output_seg = os.path.join(output_seg_dir, sample_name + ".vti")
    output_mesh = os.path.join(output_mesh_dir, sample_name + ".vtp")
    output_data = os.path.join(output_data_dir, sample_name + ".pkl")

    print("Copying file : ", sample_name)
    shutil.copy(sample_img, output_img)
    shutil.copy(sample_seg, output_seg)
    shutil.copy(sample_mesh, output_mesh)
    shutil.copy(sample_data, output_data)


def move_all_samples(sample_list):
    for sample_name in sample_list:
        move_sample(sample_name)


def move_all_augmentations(augmentation_list):
    for augmentation in augmentation_list:
        pattern = "*_" + str(augmentation) + ".vti"
        pattern_path = os.path.join(input_img_dir, pattern)
        filenames = glob.glob(pattern_path)
        filenames = [os.path.basename(fn).split(".")[0] for fn in filenames]
        move_all_samples(filenames)


input_dir = "/global/scratch/users/arjunnarayanan/WholeHeartData/cropped/mr/train/augment20/processed/"
output_dir = "/global/scratch/users/arjunnarayanan/WholeHeart-ct-crop-mr/train"
augmentations_list = range(10)

input_img_dir = os.path.join(input_dir, "vtk_image")
input_seg_dir = os.path.join(input_dir, "vtk_segmentation")
input_mesh_dir = os.path.join(input_dir, "vtk_mesh")
input_data_dir = os.path.join(input_dir, "data")

assert os.path.isdir(input_img_dir)
assert os.path.isdir(input_seg_dir)
assert os.path.isdir(input_mesh_dir)
assert os.path.isdir(input_data_dir)

output_img_dir = os.path.join(output_dir, "vtk_image")
output_seg_dir = os.path.join(output_dir, "vtk_segmentation")
output_mesh_dir = os.path.join(output_dir, "vtk_mesh")
output_data_dir = os.path.join(output_dir, "data")

make_output_directory(output_img_dir)
make_output_directory(output_seg_dir)
make_output_directory(output_mesh_dir)
make_output_directory(output_data_dir)

move_all_augmentations(augmentations_list)
