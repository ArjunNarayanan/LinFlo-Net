import glob
import pandas as pd
import os
import argparse


def verify_image_and_mesh_files(root_dir):
    image_fn = os.path.join(root_dir, "vtk_image")
    seg_fn = os.path.join(root_dir, "vtk_segmentation")
    mesh_fn = os.path.join(root_dir, "vtk_mesh")
    assert os.path.isdir(image_fn)
    assert os.path.isdir(seg_fn)
    assert os.path.isdir(mesh_fn)

    image_files = glob.glob(os.path.join(image_fn, "*.vti"))
    seg_files = glob.glob(os.path.join(seg_fn, "*.vti"))
    mesh_files = glob.glob(os.path.join(mesh_fn, "*.vtp"))
    assert len(image_files) == len(seg_files) == len(mesh_files)

    print("Verifying vtk_segmentation and vtk_mesh files for each vtk_image\n")
    for fn in image_files:
        file_name_no_ext = os.path.basename(fn).split(".")[0]
        mesh_file = os.path.join(mesh_fn, file_name_no_ext + ".vtp")
        seg_file = os.path.join(seg_fn, file_name_no_ext + ".vti")
        if not os.path.isfile(mesh_file):
            print("Missing vtk_mesh file for : ", file_name_no_ext)
        if not os.path.isfile(seg_file):
            print("Missing vtk_segmentation fole for : ", file_name_no_ext)


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
