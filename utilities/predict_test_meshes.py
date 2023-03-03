import torch
import numpy as np
import SimpleITK as sitk
import src.pre_process as pre
import os
import vtk_utils.vtk_utils as vtu
from src.template import Template
import pandas as pd
import yaml
import argparse


class Prediction:
    def __init__(self, info, model, mesh_tmplt, out_dir, modality):
        self.info = info
        self.model = model
        self.mesh_tmplt = mesh_tmplt
        self.out_dir = out_dir
        self.modality = modality

    def set_image_info(self, image_fn):
        self.image_fn = image_fn
        self.original_image = sitk.ReadImage(self.image_fn)
        self.origin = np.array(self.original_image.GetOrigin())
        self.img_center = np.array(
            self.original_image.TransformContinuousIndexToPhysicalPoint(np.array(self.original_image.GetSize()) / 2.0))
        
        template_size = self.info["input_size"]
        self.image_vol = pre.resample_spacing(self.original_image, template_size=template_size, order=1)[0]

        self.img_center2 = np.array(
            self.image_vol.TransformContinuousIndexToPhysicalPoint(np.array(self.image_vol.GetSize()) / 2.0))
        self.prediction = None

    def scale_to_image_coordinates(self, coords):
        assert coords.ndim == 2
        assert coords.shape[1] == 3

        image_size = self.info["input_size"]
        transform = vtu.build_transform_matrix(self.image_vol)
        coords = coords * np.array(image_size)
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
        self.prediction = self.mesh_tmplt.update_packed(dc).to_vtk_mesh()

    def mesh_to_segmentation(self):
        ref_img, _ = vtu.exportSitk2VTK(self.original_image)
        self.segmentation = vtu.multiclass_convert_polydata_to_imagedata(self.prediction, ref_img)


def write_all_meshes(root_dir, extension, index, prediction, output_extension):
    for filename in index["file_name"]:
        image_fn = os.path.join(root_dir, "image", filename + extension)
        assert os.path.isfile(image_fn), "Did not find file " + image_fn

        prediction.set_image_info(image_fn)
        prediction.predict_mesh()

        mesh_fn = os.path.join(prediction.out_dir, "meshes", filename + ".vtp")
        vtu.write_vtk_polydata(prediction.prediction, mesh_fn)

        prediction.mesh_to_segmentation()

        if output_extension == ".vti":
            seg_fn = os.path.join(prediction.out_dir, "segmentation", filename + ".vti")
            vtu.write_vtk_image(prediction.segmentation, seg_fn)
        elif output_extension == ".nii.gz":
            seg_fn = os.path.join(prediction.out_dir, "segmentation", filename + ".nii.gz")
            ref_im, M = vtu.exportSitk2VTK(prediction.original_image)
            print("Writing nifti with name: ", seg_fn)
            vtu.vtk_write_mask_as_nifty(prediction.segmentation, M, prediction.image_fn, seg_fn)
        else:
            raise ValueError("Unexpected output format type : ", output_extension)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict meshes on ground truth images")
    parser.add_argument("-config", help="path to config file")
    args = parser.parse_args()

    config_fn = args.config
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)

    info = config["info"]
    model_fn = config["files"]["model"]
    model = torch.load(model_fn, map_location=torch.device("cpu"))["model"]

    model.eval()

    template_fn = config["files"]["template"]
    faceids_name = config["files"].get("faceids_name", None)
    template = Template.from_vtk(template_fn, faceids_name=faceids_name)
    root_dir = config["files"]["root_dir"]
    extension = config["files"]["extension"]
    output_extension = config["files"]["output_extension"]

    default_out_dir = os.path.join(os.path.dirname(model_fn), "evaluation")
    out_dir = config["files"].get("output_dir", default_out_dir)

    meshes_dir = os.path.join(out_dir, "meshes")
    if not os.path.isdir(meshes_dir):
        os.makedirs(meshes_dir)
    seg_dir = os.path.join(out_dir, "segmentation")
    if not os.path.isdir(seg_dir):
        os.makedirs(seg_dir)

    index_fn = os.path.join(root_dir, "index.csv")
    assert os.path.isfile(index_fn), "Did not find index file at " + index_fn

    index = pd.read_csv(index_fn)

    modality = config["modality"]
    prediction = Prediction(info, model, template, out_dir, modality)

    write_all_meshes(root_dir, extension, index, prediction, output_extension)