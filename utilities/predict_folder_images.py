"""Deprecated: use ``linflonet predict --folder ...`` instead. See docs/quick_start.md."""

import os
import argparse
import warnings

from linflonet.predict import (
    PredictionConfig,
    filename_stem,
    find_image_files,
    predict_images,
)


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

    image_files = find_image_files(folder, args.extension)
    if int(args.n) >= 0:
        image_files = image_files[: int(args.n)]
    assert image_files, "Did not find any images in " + folder

    pred_config = PredictionConfig.from_yaml(args.config)
    out_dir = args.output_dir or folder
    predict_images(pred_config, image_files, out_dir, extension=args.extension)


if __name__ == "__main__":
    warnings.warn(
        "utilities/predict_folder_images.py is deprecated; use "
        "'linflonet predict --folder ...' instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
