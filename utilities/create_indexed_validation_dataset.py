import sys
import os
import argparse
import pandas as pd
import shutil


def write_output_files(root_dir, filename, out_filename, out_dir):
    print("Processing file : ", filename)

    im_fn = os.path.join(root_dir, "vtk_image", filename + ".vti")
    seg_fn = os.path.join(root_dir, "vtk_segmentation", filename + ".vti")
    mesh_fn = os.path.join(root_dir, "vtk_mesh", filename + ".vtp")
    data_fn = os.path.join(root_dir, "data", filename + ".pkl")

    out_im_fn = os.path.join(out_dir, "vtk_image", out_filename + ".vti")
    out_seg_fn = os.path.join(out_dir, "vtk_segmentation", out_filename + ".vti")
    out_mesh_fn = os.path.join(out_dir, "vtk_mesh", out_filename + ".vtp")
    out_data_fn = os.path.join(out_dir, "data", out_filename + ".pkl")

    shutil.copy(im_fn, out_im_fn)
    shutil.copy(seg_fn, out_seg_fn)
    shutil.copy(mesh_fn, out_mesh_fn)
    shutil.copy(data_fn, out_data_fn)


def write_all_output_files(root_dir, out_dir):
    index_file = os.path.join(root_dir, "index.csv")
    index = pd.read_csv(index_file)

    out_filenames = []

    for (idx, filename) in enumerate(index["file_name"]):
        out_filename = "sample{:02d}".format(idx)
        write_output_files(root_dir, filename, out_filename, out_dir)
        out_filenames.append(out_filename)

    df = pd.DataFrame(out_filenames, columns=["file_name"])
    df.sort_values(by="file_name", inplace=True)

    out_csv_file = os.path.join(out_dir, "index.csv")
    df.to_csv(out_csv_file, index=False)


def make_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset into indexed dataset for easier IO")
    parser.add_argument("-f", help="path to input folder")
    parser.add_argument("-o", help="path to output folder")
    args = parser.parse_args()

    input_folder = args.f
    output_folder = args.o

    out_im_folder = os.path.join(output_folder, "vtk_image")
    out_seg_folder = os.path.join(output_folder, "vtk_segmentation")
    out_mesh_folder = os.path.join(output_folder, "vtk_mesh")
    out_data_folder = os.path.join(output_folder, "data")

    make_dir(out_im_folder)
    make_dir(out_seg_folder)
    make_dir(out_mesh_folder)
    make_dir(out_data_folder)

    write_all_output_files(input_folder, output_folder)
