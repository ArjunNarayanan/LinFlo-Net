import numpy as np
import os
import SimpleITK as sitk
import glob
import argparse
import sys

sys.path.append(os.getcwd())
import vtk_utils.vtk_utils as vtu


def vtk_to_numpy(img):
    x, y, z = img.GetDimensions()
    py_img = vtu.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
    return py_img


def load_image(filename):
    img = vtu.load_vtk_image(filename)
    return vtk_to_numpy(img)


def make_output_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", required=True)
    parser.add_argument("-output", required=True)
    parser.add_argument("-ext", default=".vti")
    args = parser.parse_args()

    input_folder = args.input
    output_folder = args.output
    make_output_dir(output_folder)
    input_ext = args.ext

    filepaths = glob.glob(os.path.join(input_folder, "*" + input_ext))

    for input_file in filepaths:
        print("Processing : ", input_file)
        # input_file = os.path.join(input_folder, filename)

        arr = load_image(input_file)
        sitk_img = sitk.GetImageFromArray(arr)

        filename = os.path.basename(input_file)
        output_file = os.path.splitext(filename)[0] + ".nii.gz"
        output_filepath = os.path.join(output_folder, output_file)
        print("Writing output : ", output_filepath, "\n")
        sitk.WriteImage(sitk_img, output_filepath)
