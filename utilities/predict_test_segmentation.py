import torch
import numpy as np
import SimpleITK as sitk
import os
import sys
import pandas as pd
import argparse

sys.path.append(os.getcwd())
import src.pre_process as pre
from src.io_utils import load_yaml_config

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class ProcessedImage:
    def __init__(self, image_fn, modality, normalized_size):
        self.image_fn = image_fn
        self.modality = modality
        self.normalized_size = normalized_size
        self.original_image = sitk.ReadImage(image_fn)
        self.normalized_image, _ = pre.resample_spacing(self.original_image, template_size=3 * [normalized_size],
                                                        order=1)
        self.model_input = self.get_model_input(self.normalized_image, modality)

    @staticmethod
    def get_image_center(sitk_img):
        center_idx = np.array(sitk_img.GetSize()) / 2.0
        center_coords = np.array(sitk_img.TransformContinuousIndexToPhysicalPoint(center_idx))
        return center_coords

    @staticmethod
    def get_model_input(normalized_image, modality):
        img_vol = sitk.GetArrayFromImage(normalized_image).transpose(2, 1, 0)
        img_vol = pre.rescale_intensity(img_vol, modality, [750, -750])
        torch_img = torch.tensor(img_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return torch_img


def predict_segmentation(model, image):
    with torch.no_grad():
        seg = model.get_segmentation(image)

    seg = seg.squeeze(0)
    seg = seg.argmax(dim=0)
    return seg


def transform_segmentation_to_image_space(seg, ref_img):
    transformed = pre.centering(seg, ref_img, order=0)
    return transformed


def numpy_to_itk(array, ref_img):
    img = sitk.GetImageFromArray(array.transpose(2, 1, 0))
    img.CopyInformation(ref_img)
    return img


def get_segmentation_in_image_space(model, processed_image):
    normalized_prediction = predict_segmentation(model, processed_image.model_input)
    normalized_prediction = normalized_prediction.cpu().numpy().astype('int16')
    normalized_sitk_img = numpy_to_itk(normalized_prediction, processed_image.normalized_image)
    sitk_img = transform_segmentation_to_image_space(normalized_sitk_img, processed_image.original_image)
    return sitk_img


def write_all_segmentations(root_dir, index, model, output_dir, modality, normalized_image_size):
    for filename in index["file_name"]:
        image_fn = os.path.join(root_dir, "image", filename + ".nii.gz")
        assert os.path.isfile(image_fn)

        processed_image = ProcessedImage(image_fn, modality, normalized_image_size)
        torch_seg = predict_segmentation(model, processed_image.model_input)
        np_seg = torch_seg.cpu().numpy()
        np_seg = np_seg.astype("int16")
        sitk_img = numpy_to_itk(np_seg, processed_image.original_image)

        output_filename = os.path.join(output_dir, filename + ".nii.gz")
        sitk.WriteImage(sitk_img, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model Dice Scores")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    input_dir = config["input_dir"]
    model_fn = config["model"]
    output_dir = config["output_dir"]
    image_size = config["normalized_image_size"]
    modality = config["modality"]

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model_data = torch.load(model_fn, map_location=torch.device("cpu"))
    model = model_data["model"]

    index_fn = os.path.join(input_dir, "index.csv")
    assert os.path.isfile(index_fn)
    index = pd.read_csv(index_fn)

    write_all_segmentations(input_dir, index, model, output_dir, modality, image_size)
