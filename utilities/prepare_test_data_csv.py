import glob
import pandas as pd
import os
import argparse


def make_file_index(root_dir, num_files, extension):
    assert os.path.isdir(os.path.join(root_dir, "image"))

    pattern = "image/*" + extension
    image_files = glob.glob(os.path.join(root_dir, pattern))

    if num_files >= 0:
        image_files = image_files[:num_files]

    file_names = [os.path.basename(f) for f in image_files]
    file_name_no_ext = [f.split(".")[0] for f in file_names]

    df = pd.DataFrame(file_name_no_ext, columns=["file_name"])
    df.sort_values(by="file_name",inplace=True)

    print("Found ", len(df), " files,")
    outfile = os.path.join(root_dir, "index.csv")
    print("Writing index file at ", outfile, "\n")
    df.to_csv(outfile, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare File Index")
    parser.add_argument("-f", "--folder", help="Folder to process")
    parser.add_argument("-n", default=-1, help="Number of files to process")
    parser.add_argument("-e", help = "File extension")
    args = parser.parse_args()

    make_file_index(args.folder, int(args.n), args.e)
