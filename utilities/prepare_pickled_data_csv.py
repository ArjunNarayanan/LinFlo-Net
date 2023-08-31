import glob
import pandas as pd
import os
import argparse


def make_file_index(root_dir, data_dir, num_files):

    data_files = glob.glob(os.path.join(root_dir, data_dir, "*.pkl"))

    if num_files >= 0:
        data_files = data_files[:num_files]

    file_names = [os.path.basename(f) for f in data_files]
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
    parser.add_argument("-data", help="data folder", default="data")
    parser.add_argument("-n", default=-1, help="Number of files to process")

    args = parser.parse_args()
    
    make_file_index(args.folder, args.data, int(args.n))
