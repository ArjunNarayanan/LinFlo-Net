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


def crop_and_write(file_idx):
    file_name = dataframe.loc[file_idx, "sample_name"]
    file_path = os.path.join(input_dir, file_name + ".nii.gz")
    assert os.path.isfile(file_path)
    img = sitk.ReadImage(file_path)

    lower_bound = dataframe.loc[file_idx, ["X0", "Y0", "Z0"]]
    upper_bound = dataframe.loc[file_idx, ["X1", "Y1", "Z1"]]
    img_size = dataframe.loc[file_idx, ["Xm", "Ym", "Zm"]]

    assert all(s1 + 1 == s2 for (s1, s2) in zip(img_size, img.GetSize()))

    cropped_image = img[lower_bound[0]:upper_bound[0],
                    lower_bound[1]:upper_bound[1],
                    lower_bound[2]:upper_bound[2]]

    output_path = os.path.join(output_dir, file_name + ".nii.gz")
    print("Writing output at : ", output_path)
    sitk.WriteImage(cropped_image, output_path)


def crop_all_images_segmentations():
    num_files = len(dataframe)
    for idx in range(num_files):
        crop_and_write(idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop images and segmentations based on ROI")
    parser.add_argument("-f", help="Input folder", required=True)
    parser.add_argument("-c", help="Path to CSV file", required=True)
    parser.add_argument("-o", help="Output folder", required=True)

    args = parser.parse_args()

    input_dir = args.f
    assert os.path.isdir(input_dir)

    output_dir = args.o
    create_output_folders(output_dir)

    index_file = args.c
    assert os.path.isfile(index_file)
    dataframe = pd.read_csv(index_file)

    crop_all_images_segmentations()
