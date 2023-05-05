import glob
import pandas as pd
import os
import argparse


def verify_image_and_mesh_files(root_dir, extension):
    image_fn = os.path.join(root_dir, "image")
    mesh_fn = os.path.join(root_dir, "label")
    assert os.path.isdir(image_fn)
    assert os.path.isdir(mesh_fn)

    image_files = glob.glob(os.path.join(root_dir, "image", "*" + extension))

    print("Verifying label files for each image\n")
    for fn in image_files:
        file_name = os.path.basename(fn)
        seg_file = os.path.join(root_dir, "label", file_name)
        if not os.path.isfile(seg_file):
            print("Missing label file for : ", file_name)

    seg_files = glob.glob(os.path.join(root_dir, "label", "*" + extension))

    print("Verifying image files for each label\n")
    for fn in seg_files:
        file_name = os.path.basename(fn)
        image_file = os.path.join(root_dir, "image", file_name)
        if not os.path.isfile(image_file):
            print("Missing image file for : ", file_name)


def make_file_index(root_dir, num_files, extension):
    assert os.path.isdir(os.path.join(root_dir, "image"))

    pattern = "image/*" + extension
    image_files = glob.glob(os.path.join(root_dir, pattern))

    if num_files >= 0:
        image_files = image_files[:num_files]

    file_names = [os.path.basename(f) for f in image_files]
    file_name_no_ext = [f.split(".")[0] for f in file_names]

    df = pd.DataFrame(file_name_no_ext, columns=["file_name"])
    df.sort_values(by="file_name", inplace=True)

    print("Found ", len(df), " files,")
    outfile = os.path.join(root_dir, "index.csv")
    print("Writing index file at ", outfile, "\n")
    df.to_csv(outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare File Index")
    parser.add_argument("-f", "--folder", help="Folder to process", required=True)
    parser.add_argument("-n", default=-1, help="Number of files to process")
    parser.add_argument("-e", help="File extension", default = ".nii.gz")
    args = parser.parse_args()

    verify_image_and_mesh_files(args.folder, args.e)
    make_file_index(args.folder, int(args.n), args.e)
