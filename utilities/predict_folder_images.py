import os
import sys
import yaml
import argparse

sys.path.append(os.getcwd())
from utilities.prepare_test_data_csv import make_file_index, filename_stem
from utilities.predict_test_meshes import create_prediction, write_one_mesh


def main():
    parser = argparse.ArgumentParser(description="Predict meshes and segmentations for all images in a folder")
    parser.add_argument("-config", help="path to config file", required=True)
    parser.add_argument("-f", "--folder", help="folder containing images or an image/ subdirectory", required=True)
    parser.add_argument("-e", "--extension", help="input image file extension", default=".nii.gz")
    parser.add_argument("-n", default=-1, help="number of files to process")
    parser.add_argument("-o", "--output_dir", help="output directory (overrides config)", default=None)
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    assert os.path.isdir(folder), "Did not find folder " + folder

    image_files, _ = make_file_index(folder, int(args.n), args.extension)

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    out_dir = args.output_dir or folder
    prediction, output_extension = create_prediction(config, out_dir)

    for image_fn in image_files:
        filename = filename_stem(image_fn, args.extension)
        print("Processing file : ", filename)
        write_one_mesh(prediction, image_fn, filename, output_extension)


if __name__ == "__main__":
    main()
