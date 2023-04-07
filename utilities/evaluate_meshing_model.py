import torch
import numpy as np
import SimpleITK as sitk
import os
import sys

sys.path.append(os.getcwd())
import src.pre_process as pre
import vtk_utils.vtk_utils as vtu
from src.template import Template
from src.utilities import dice_score
import pandas as pd
import yaml
import argparse


class Prediction:
    def __init__(self, model, mesh_tmplt, out_dir, modality):
        self.model = model
        self.mesh_tmplt = mesh_tmplt
        self.out_dir = out_dir
        self.modality = modality

    def set_image_info(self, image_fn, gt_seg_fn, size):
        self.image_fn = image_fn
        self.gt_seg_fn = gt_seg_fn
        self.gt_seg = sitk.ReadImage(self.gt_seg_fn)
        self.image_vol = sitk.ReadImage(self.image_fn)
        self.origin = np.array(self.image_vol.GetOrigin())
        self.img_center = np.array(
            self.image_vol.TransformContinuousIndexToPhysicalPoint(np.array(self.image_vol.GetSize()) / 2.0))
        self.size = size
        self.image_vol = pre.resample_spacing(self.image_vol, template_size=size, order=1)[0]

        self.img_center2 = np.array(
            self.image_vol.TransformContinuousIndexToPhysicalPoint(np.array(self.image_vol.GetSize()) / 2.0))
        self.prediction = None

    def scale_to_image_coordinates(self, coords):
        assert coords.ndim == 2
        assert coords.shape[1] == 3

        transform = vtu.build_transform_matrix(self.image_vol)
        coords = coords * np.array(self.size)
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)
        coords = np.matmul(transform, coords.T).T[:, :3]
        coords = coords + self.img_center - self.img_center2
        return coords

    def get_torch_image(self):
        img_vol = sitk.GetArrayFromImage(self.image_vol).transpose(2, 1, 0)
        img_vol = pre.rescale_intensity(img_vol, self.modality, [750, -750])
        self.original_shape = img_vol.shape
        torch_img = torch.tensor(img_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return torch_img

    def predict_mesh(self):
        template_coords = self.mesh_tmplt.verts_packed().unsqueeze(0)
        torch_img = self.get_torch_image()

        with torch.no_grad():
            deformed_coords = self.model(torch_img, template_coords)

        deformed_coords = deformed_coords.squeeze(0).numpy()
        deformed_coords = self.scale_to_image_coordinates(deformed_coords)

        dc = torch.tensor(deformed_coords, dtype=torch.float32)
        self.prediction = template.update_packed(dc).to_vtk_mesh()

    def mesh_to_segmentation(self):
        ref_img, _ = vtu.exportSitk2VTK(self.gt_seg)
        self.segmentation = vtu.multiclass_convert_polydata_to_imagedata(self.prediction, ref_img)

    def evaluate_dice(self):
        print("Evaluating dice: ", self.image_fn)
        ref_im = sitk.ReadImage(self.gt_seg_fn)
        ref_im, M = vtu.exportSitk2VTK(ref_im)
        ref_im_py = pre.swapLabels_ori(vtu.vtk_to_numpy(ref_im.GetPointData().GetScalars()))
        pred_im_py = vtu.vtk_to_numpy(self.segmentation.GetPointData().GetScalars())
        dice_values = dice_score(pred_im_py, ref_im_py)
        return dice_values


def compute_all_dice_scores(root_dir, index, prediction, size=(128, 128, 128)):
    dice = []
    for filename in index["file_name"]:
        image_fn = os.path.join(root_dir, "image", filename + ".nii")
        gt_seg_fn = os.path.join(root_dir, "label", filename + ".nii")
        prediction.set_image_info(image_fn, gt_seg_fn, size)
        prediction.predict_mesh()

        mesh_fn = os.path.join(prediction.out_dir, "meshes", filename + ".vtp")
        vtu.write_vtk_polydata(prediction.prediction, mesh_fn)

        prediction.mesh_to_segmentation()

        seg_fn = os.path.join(prediction.out_dir, "segmentation", filename + ".vti")
        # gt_seg_vtk, _ = vtu.exportSitk2VTK(prediction.gt_seg)
        vtu.write_vtk_image(prediction.segmentation, seg_fn)

        dice.append(prediction.evaluate_dice())

    return dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model Dice Scores")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)

    model_fn = config["files"]["model"]
    print("LOADING MODEL AT : ", model_fn, "\n\n")

    model = torch.load(model_fn, map_location=torch.device("cpu"))["model"]
    model.eval()

    template_fn = config["files"]["template"]
    template = Template.from_vtk(template_fn)
    root_dir = config["files"]["root_dir"]
    assert os.path.isdir(root_dir)

    out_dir = config["files"]["out_dir"]

    meshes_dir = os.path.join(out_dir, "meshes")
    if not os.path.isdir(meshes_dir):
        os.makedirs(meshes_dir)
    seg_dir = os.path.join(out_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)

    index_fn = os.path.join(root_dir, "index.csv")
    index = pd.read_csv(index_fn)

    modality = config["modality"]
    prediction = Prediction(model, template, out_dir, modality)

    dice = compute_all_dice_scores(root_dir, index, prediction)

    dice = np.array(dice)
    column_names = ["Background", "Left Ventricle", "Left Atrium", "LV Blood Pool", "Right Atrium", "Right Ventricle",
                    "Aorta", "Pulmonary Artery"]
    df = pd.DataFrame(dice, columns=column_names)
    df.index = index["file_name"]

    out_file = os.path.join(config["files"]["out_dir"], "dice.csv")
    df.to_csv(out_file)
