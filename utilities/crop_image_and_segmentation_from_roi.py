import SimpleITK as sitk
import pandas as pd
import os
import shutil
import argparse


def create_output_folders(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    else:
        print("Warning: deleting existing output folder!")
        shutil.rmtree(folder)
        os.makedirs(folder)


def crop_and_write_image_segmentation(file_idx):
    fn = dataframe.loc[file_idx, "sample_name"]
    print("Processing file : ", fn)

    img_file = os.path.join(input_img_dir, fn + ".nii.gz")
    seg_file = os.path.join(input_seg_dir, fn + ".nii.gz")
    assert os.path.isfile(img_file)
    assert os.path.isfile(seg_file)

    img = sitk.ReadImage(img_file)
    seg = sitk.ReadImage(seg_file)

    lower_bound = dataframe.loc[file_idx, ["X0", "Y0", "Z0"]]
    upper_bound = dataframe.loc[file_idx, ["X1", "Y1", "Z1"]]

    # print("Lower bound : ", lower_bound)
    # print("Upper bound : ", upper_bound)

    cropped_image = img[lower_bound[0]:upper_bound[0],
                    lower_bound[1]:upper_bound[1],
                    lower_bound[2]:upper_bound[2]]
    cropped_seg = seg[lower_bound[0]:upper_bound[0],
                  lower_bound[1]:upper_bound[1],
                  lower_bound[2]:upper_bound[2]]

    output_img_file = os.path.join(output_img_dir, fn + ".nii.gz")
    output_seg_file = os.path.join(output_seg_dir, fn + ".nii.gz")

    print("Writing output for file ", fn)
    sitk.WriteImage(cropped_image, output_img_file)
    sitk.WriteImage(cropped_seg, output_seg_file)


def crop_all_images_segmentations():
    num_files = len(dataframe)
    for idx in range(num_files):
        crop_and_write_image_segmentation(idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images and segmentations based on ROI")
    parser.add_argument("-f", help="folder to process")
    parser.add_argument("-c", help="csv file containing crop information")
    parser.add_argument("-o", help="output directory name")
    args = parser.parse_args()

    base_dir = args.f
    input_img_dir = os.path.join(base_dir, "image")
    input_seg_dir = os.path.join(base_dir, "label")
    assert os.path.isdir(input_img_dir)
    assert os.path.isdir(input_seg_dir)

    output_dir = os.path.join(base_dir, args.o)
    output_img_dir = os.path.join(output_dir, "image")
    output_seg_dir = os.path.join(output_dir, "label")
    create_output_folders(output_img_dir)
    create_output_folders(output_seg_dir)

    csv_file = args.c
    index_file = os.path.join(base_dir, csv_file)
    assert os.path.isfile(index_file)
    dataframe = pd.read_csv(index_file)

    crop_all_images_segmentations()
