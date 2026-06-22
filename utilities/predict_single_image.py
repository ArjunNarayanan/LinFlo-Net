import os
import argparse

from linflonet.predict import (
    PredictionConfig,
    create_prediction,
    filename_stem,
    find_image_files,
    write_one_mesh,
)


def main():
    parser = argparse.ArgumentParser(description="Predict a mesh and segmentation for a single image")
    parser.add_argument("-config", help="path to config file", required=True)
    parser.add_argument("-i", "--image", help="path to a single input image", required=True)
    parser.add_argument("-e", "--extension", help="input image file extension", default=".nii.gz")
    parser.add_argument("-o", "--output_dir", help="output directory (overrides config)", default=None)
    args = parser.parse_args()

    image_fn = args.image
    assert os.path.isfile(image_fn), "Did not find image file " + image_fn

    pred_config = PredictionConfig.from_yaml(args.config)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(image_fn))
    prediction = create_prediction(pred_config, out_dir)

    filename = filename_stem(image_fn, args.extension)
    write_one_mesh(prediction, image_fn, filename, pred_config.output_extension)


if __name__ == "__main__":
    main()
