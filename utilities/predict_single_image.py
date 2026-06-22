import os
import sys
import yaml
import argparse

sys.path.append(os.getcwd())
from utilities.prepare_test_data_csv import filename_stem
from utilities.predict_test_meshes import create_prediction, write_one_mesh


def main():
    parser = argparse.ArgumentParser(description="Predict a mesh and segmentation for a single image")
    parser.add_argument("-config", help="path to config file", required=True)
    parser.add_argument("-i", "--image", help="path to a single input image", required=True)
    parser.add_argument("-e", "--extension", help="input image file extension", default=".nii.gz")
    parser.add_argument("-o", "--output_dir", help="output directory (overrides config)", default=None)
    args = parser.parse_args()

    image_fn = args.image
    assert os.path.isfile(image_fn), "Did not find image file " + image_fn

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(image_fn))
    prediction, output_extension = create_prediction(config, out_dir)

    filename = filename_stem(image_fn, args.extension)
    write_one_mesh(prediction, image_fn, filename, output_extension)


if __name__ == "__main__":
    main()
