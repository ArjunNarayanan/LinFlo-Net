import os
import argparse
import pandas as pd
import shutil


def write_output_files(root_dir, filename, out_filename, out_dir):
    print("Processing file : ", filename)

    im_fn = os.path.join(root_dir, "image", filename + ".nii.gz")
    seg_fn = os.path.join(root_dir, "label", filename + ".nii.gz")
    mesh_fn = os.path.join(root_dir, "meshes", filename + ".vtp")

    out_im_fn = os.path.join(out_dir, "image", out_filename + ".nii.gz")
    out_seg_fn = os.path.join(out_dir, "label", out_filename + ".nii.gz")
    out_mesh_fn = os.path.join(out_dir, "meshes", out_filename + ".vtp")

    shutil.copy(im_fn, out_im_fn)
    shutil.copy(seg_fn, out_seg_fn)
    shutil.copy(mesh_fn, out_mesh_fn)


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

    assert os.path.isdir(os.path.join(input_folder, "image"))
    assert os.path.isdir(os.path.join(input_folder, "label"))
    assert os.path.isdir(os.path.join(input_folder, "meshes"))

    out_im_folder = os.path.join(output_folder, "image")
    out_seg_folder = os.path.join(output_folder, "label")
    out_mesh_folder = os.path.join(output_folder, "meshes")

    make_dir(out_im_folder)
    make_dir(out_seg_folder)
    make_dir(out_mesh_folder)

    write_all_output_files(input_folder, output_folder)
