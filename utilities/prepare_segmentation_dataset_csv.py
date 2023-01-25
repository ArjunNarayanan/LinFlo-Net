import glob
import pandas as pd
import os
import argparse


def verify_image_and_mesh_files(root_dir):
    image_fn = os.path.join(root_dir, "vtk_image")
    mesh_fn = os.path.join(root_dir, "vtk_segmentation")
    assert os.path.isdir(image_fn)
    assert os.path.isdir(mesh_fn)

    image_files = glob.glob(os.path.join(root_dir, "vtk_image/*.vti"))

    print("Verifying vtk_segmentation files for each vtk_image\n")
    for fn in image_files:
        file_name_no_ext = os.path.basename(fn).split(".")[0]
        mesh_file = os.path.join(root_dir, "vtk_segmentation", file_name_no_ext + ".vti")
        if not os.path.isfile(mesh_file):
            print("Missing vtk_mesh file for : ", file_name_no_ext)

    mesh_files = glob.glob(os.path.join(root_dir, "vtk_segmentation/*.vti"))

    print("Verifying vtk_image files for each vtk_segmentation\n")
    for fn in mesh_files:
        file_name_no_ext = os.path.basename(fn).split(".")[0]
        image_file = os.path.join(root_dir, "vtk_image", file_name_no_ext + ".vti")
        if not os.path.isfile(image_file):
            print("Missing vtk_image file for : ", file_name_no_ext)


def make_file_index(root_dir, num_files):
    image_files = glob.glob(os.path.join(root_dir, "vtk_image/*.vti"))

    if num_files >= 0:
        image_files = image_files[:num_files]

    file_names = [os.path.basename(f) for f in image_files]
    file_name_no_ext = [f.split(".")[0] for f in file_names]

    df = pd.DataFrame(file_name_no_ext, columns=["file_name"])
    df.sort_values(by="file_name", inplace=True)

    num_files = len(df)
    print("Found ", num_files, " files")

    outfile = os.path.join(root_dir, "index.csv")
    print("Writing index file at ", outfile, "\n")
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare File Index")
    parser.add_argument("-f", "--folder", help="Folder to process")
    parser.add_argument("-n", default=-1, help="Number of files to process")

    args = parser.parse_args()
    verify_image_and_mesh_files(args.folder)
    make_file_index(args.folder, int(args.n))
