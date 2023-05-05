import os
import SimpleITK as sitk
import argparse
import glob
import sys

sys.path.append(os.getcwd())
import vtk_utils.vtk_utils as vtu


def convert_and_save(filename, input_dir, output_dir, input_extension, output_extension):
    filepath = os.path.join(input_dir, filename + input_extension)
    sitk_img = sitk.ReadImage(filepath)
    vtk_img, _ = vtu.exportSitk2VTK(sitk_img)
    out_file = os.path.join(output_dir, filename + output_extension)
    vtu.write_vtk_image(vtk_img, out_file)


def convert_all_files(input_dir, output_dir, filename_index, input_extension, output_extension):
    for filename in filename_index:
        print("Converting file : ", filename)
        convert_and_save(filename, input_dir, output_dir, input_extension, output_extension)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert nifti files to vtk files")
    parser.add_argument("-f", help="folder to process")
    parser.add_argument("-o", help="output folder")
    args = parser.parse_args()

    input_dir = args.f
    output_dir = args.o

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    input_extension = ".nii.gz"
    output_extension = ".vti"

    filenames = glob.glob(os.path.join(input_dir, "*" + input_extension))
    filenames = [os.path.basename(fn) for fn in filenames]
    filenames = [fn.split(".")[0] for fn in filenames]

    convert_all_files(input_dir, output_dir, filenames, input_extension, output_extension)