import os
import sys
import SimpleITK as sitk
import glob
import numpy as np
import pandas as pd
import argparse

sys.path.append(os.getcwd())
from src.utilities import dice_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-prediction", help="folder with segmentation predictions", required=True)
    parser.add_argument("-reference", help="folder with ground truth data", required=True)
    parser.add_argument("-extension", help="filename extension", default=".nii.gz")
    parser.add_argument("-output", help="output file name", default="dice.csv")
    args = parser.parse_args()

    pred_folder = args.prediction
    gt_folder = args.reference
    extension = args.extension
    output_filename = args.output

    output_file = os.path.join(pred_folder, output_filename)

    column_names = ["Background", "Myocardium", "Left Atrium", "LV Blood Pool", "Right Atrium", "Right Ventricle",
                    "Aorta", "Pulmonary Artery"]

    filenames = glob.glob(os.path.join(gt_folder, "*" + extension))
    dice_results = []
    sample_names = []

    for gt_file in filenames:
        filename = os.path.basename(gt_file)
        print("\tProcessing file : ", filename)

        pred_file = os.path.join(pred_folder, filename)

        assert os.path.isfile(pred_file)

        pred_seg = sitk.ReadImage(pred_file)
        gt_seg = sitk.ReadImage(gt_file)

        pred_arr = sitk.GetArrayFromImage(pred_seg)
        gt_arr = sitk.GetArrayFromImage(gt_seg)

        dice_values = dice_score(pred_arr, gt_arr)
        dice_results.append(dice_values)
        sample_names.append(filename.split(".")[0])

    dice_data = np.array(dice_results)
    df = pd.DataFrame(dice_data, columns=column_names)
    df.index = sample_names

    print("Writing output file : ", output_file)
    df.to_csv(output_file)
