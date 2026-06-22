import glob
import pandas as pd
import os
import argparse


def find_image_files(root_dir, extension):
    image_dir = os.path.join(root_dir, "image")
    if os.path.isdir(image_dir):
        search_dir = image_dir
    else:
        search_dir = root_dir

    pattern = os.path.join(search_dir, "*" + extension)
    return sorted(glob.glob(pattern))


def filename_stem(filepath, extension):
    basename = os.path.basename(filepath)
    if extension.startswith(".") and basename.endswith(extension):
        return basename[: -len(extension)]
    return basename.split(".")[0]


def make_file_index(root_dir, num_files, extension):
    image_files = find_image_files(root_dir, extension)

    if num_files >= 0:
        image_files = image_files[:num_files]

    assert len(image_files) > 0, "Did not find any images in " + root_dir

    file_name_no_ext = [filename_stem(f, extension) for f in image_files]

    df = pd.DataFrame(file_name_no_ext, columns=["file_name"])
    df.sort_values(by="file_name", inplace=True)

    print("Found ", len(df), " files,")
    outfile = os.path.join(root_dir, "index.csv")
    print("Writing index file at ", outfile, "\n")
    df.to_csv(outfile, index=False)

    return image_files, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare File Index")
    parser.add_argument("-f", "--folder", help="Folder to process", required=True)
    parser.add_argument("-n", default=-1, help="Number of files to process")
    parser.add_argument("-e", help="File extension", default = ".nii.gz")
    args = parser.parse_args()

    make_file_index(args.folder, int(args.n), args.e)
