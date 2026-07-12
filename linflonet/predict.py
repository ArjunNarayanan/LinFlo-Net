"""Prediction API for LinFlo-Net."""

from __future__ import annotations

import glob
import inspect
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import SimpleITK as sitk
import torch
import yaml

import src.pre_process as pre
import vtk_utils.vtk_utils as vtu
from linflonet.paths import resolve_template_path
from src.io_utils import read_image
from src.template import Template

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def filename_stem(filepath: str, extension: str) -> str:
    basename = os.path.basename(filepath)
    if extension.startswith(".") and basename.endswith(extension):
        return basename[: -len(extension)]
    return basename.split(".")[0]


def find_image_files(folder: str, extension: str) -> list[str]:
    folder = os.path.abspath(folder)
    image_dir = os.path.join(folder, "image")
    search_dir = image_dir if os.path.isdir(image_dir) else folder
    pattern = os.path.join(search_dir, "*" + extension)
    return sorted(glob.glob(pattern))


@dataclass
class PredictionConfig:
    model: str
    template: str
    modality: str
    input_size: tuple[int, int, int] = (128, 128, 128)
    faceids_name: Optional[str] = None
    output_extension: str = ".nii.gz"
    root_dir: Optional[str] = None
    extension: str = ".nii.gz"
    output_dir: Optional[str] = None
    template_distance_map: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "PredictionConfig":
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        files = config["files"]

        return cls(
            model=files["model"],
            template=resolve_template_path(files["template"]),
            modality=config["modality"],
            input_size=tuple(config["info"]["input_size"]),
            faceids_name=files.get("faceids_name"),
            output_extension=files.get("output_extension", ".nii.gz"),
            root_dir=files.get("root_dir"),
            extension=files.get("extension", ".nii.gz"),
            output_dir=files.get("output_dir"),
            template_distance_map=files.get("template_distance_map"),
        )


def model_needs_distance_map(model: torch.nn.Module) -> bool:
    """Whether *model* is a UDF model whose forward expects a distance map."""
    return "distance_map" in inspect.signature(model.forward).parameters


def load_distance_map(path: str) -> torch.Tensor:
    """Load a template distance map (``.vtk`` or ``.pth``) as a 4D tensor on *device*.

    Matches the loading convention used during UDF training: VTK images are read
    into a 3D tensor and given a leading channel dim, whereas ``.pth`` tensors are
    assumed to already carry the channel dimension.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vtk":
        distance_map = read_image(path).unsqueeze(0)
    elif ext == ".pth":
        distance_map = torch.load(path, map_location=torch.device("cpu"))
    else:
        raise ValueError(
            f"Unexpected template distance map extension {ext!r}; expected .vtk or .pth"
        )
    return distance_map.to(device)


def _forward_deformed_coords(
    model: torch.nn.Module,
    torch_img: torch.Tensor,
    template_coords: torch.Tensor,
    distance_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the LT+flow (or UDF LT+flow) forward pass and return deformed coords."""
    params = inspect.signature(model.forward).parameters
    if "distance_map" in params:
        if distance_map is None:
            raise ValueError(
                f"{type(model).__name__} requires a UDF distance map. Provide "
                "files.template_distance_map in the config or --distance-map on the CLI."
            )
        return model(torch_img, template_coords, distance_map)
    if "multiplication_factor" in params:
        return model(torch_img, template_coords, multiplication_factor=1.0)
    return model(torch_img, template_coords)


class Prediction:
    def __init__(
        self,
        config: PredictionConfig,
        out_dir: str,
        model: torch.nn.Module,
        mesh_template: Template,
        distance_map: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.info = {"input_size": list(config.input_size)}
        self.model = model
        self.mesh_tmplt = mesh_template
        self.out_dir = out_dir
        self.modality = config.modality
        self.distance_map = distance_map
        self.prediction = None

    def set_image_info(self, image_fn: str) -> None:
        self.image_fn = image_fn
        self.original_image = sitk.ReadImage(self.image_fn)
        self.img_center = np.array(
            self.original_image.TransformContinuousIndexToPhysicalPoint(
                np.array(self.original_image.GetSize()) / 2.0
            )
        )

        template_size = self.info["input_size"]
        self.image_vol = pre.resample_spacing(
            self.original_image, template_size=template_size, order=1
        )[0]

        self.img_center2 = np.array(
            self.image_vol.TransformContinuousIndexToPhysicalPoint(
                np.array(self.image_vol.GetSize()) / 2.0
            )
        )
        self.prediction = None

    def scale_to_image_coordinates(self, coords: np.ndarray) -> np.ndarray:
        assert coords.ndim == 2 and coords.shape[1] == 3

        image_size = self.info["input_size"]
        transform = vtu.build_transform_matrix(self.image_vol)
        coords = coords * np.array(image_size)
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)
        coords = np.matmul(transform, coords.T).T[:, :3]
        coords = coords + self.img_center - self.img_center2
        return coords

    def get_torch_image(self) -> torch.Tensor:
        img_vol = sitk.GetArrayFromImage(self.image_vol).transpose(2, 1, 0)
        img_vol = pre.rescale_intensity(img_vol, self.modality, [750, -750])
        return torch.tensor(img_vol, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def predict_mesh(self) -> None:
        template_coords = self.mesh_tmplt.verts_packed().unsqueeze(0).to(device)
        torch_img = self.get_torch_image().to(device)

        with torch.no_grad():
            deformed_coords = _forward_deformed_coords(
                self.model, torch_img, template_coords, self.distance_map
            )

        deformed_coords = deformed_coords.squeeze(0).detach().cpu().numpy()
        deformed_coords = self.scale_to_image_coordinates(deformed_coords)

        dc = torch.tensor(deformed_coords, dtype=torch.float32)
        self.prediction = self.mesh_tmplt.update_packed(dc).to_vtk_mesh()

    def mesh_to_segmentation(self) -> None:
        ref_img, _ = vtu.exportSitk2VTK(self.original_image)
        self.segmentation = vtu.multiclass_convert_polydata_to_imagedata(
            self.prediction, ref_img
        )


def _nifti_reference_path(prediction: "Prediction") -> tuple[str, Optional[str]]:
    """Return a path to a NIfTI file describing the original image geometry.

    ``vtk_write_mask_as_nifty`` reads the source image with VTK's NIfTI reader
    purely to recover the QForm/SForm/QFac header info. That reader only
    understands NIfTI, so for other SimpleITK-readable inputs (e.g. ``.mha``)
    we write a temporary NIfTI copy of the original image and hand that over
    instead. Returns ``(reference_path, temp_path_to_cleanup)``; the second
    element is ``None`` when the original file was already NIfTI.
    """
    image_fn = prediction.image_fn
    lower = image_fn.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return image_fn, None

    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    sitk.WriteImage(prediction.original_image, tmp.name)
    return tmp.name, tmp.name


def _ensure_output_dirs(out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "segmentation"), exist_ok=True)


def create_prediction(config: PredictionConfig, out_dir: str) -> Prediction:
    model = torch.load(
        config.model, map_location=torch.device("cpu"), weights_only=False
    )["model"]
    model.to(device)
    model.eval()

    needs_distance_map = model_needs_distance_map(model)
    model_kind = "UDF linear transform + flow" if needs_distance_map else "linear transform + flow"
    print(f"Loaded model: {type(model).__name__} ({model_kind})")

    distance_map = None
    if config.template_distance_map:
        distance_map = load_distance_map(config.template_distance_map)
        print(f"Loaded template distance map: {config.template_distance_map}")

    if needs_distance_map and distance_map is None:
        raise ValueError(
            f"{type(model).__name__} is a UDF model and requires a template distance "
            "map. Provide files.template_distance_map in the config or --distance-map "
            "on the CLI (a .vtk or .pth file)."
        )
    if distance_map is not None and not needs_distance_map:
        print(
            "WARNING: a template distance map was provided but the model does not use "
            "one; ignoring it."
        )
        distance_map = None

    template = Template.from_vtk(config.template, faceids_name=config.faceids_name)

    _ensure_output_dirs(out_dir)
    return Prediction(config, out_dir, model, template, distance_map=distance_map)


def write_one_mesh(
    prediction: Prediction, image_fn: str, filename: str, output_extension: str
) -> None:
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
        _, transform = vtu.exportSitk2VTK(prediction.original_image)
        print("Writing nifti with name:", seg_fn)
        ref_path, tmp_path = _nifti_reference_path(prediction)
        try:
            vtu.vtk_write_mask_as_nifty(
                prediction.segmentation, transform, ref_path, seg_fn
            )
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        raise ValueError(f"Unexpected output format type: {output_extension}")


def predict_images(
    config: PredictionConfig,
    image_paths: Iterable[str],
    out_dir: str,
    extension: str = ".nii.gz",
) -> None:
    prediction = create_prediction(config, out_dir)
    for image_fn in image_paths:
        filename = filename_stem(image_fn, extension)
        print("Processing file:", filename)
        write_one_mesh(prediction, image_fn, filename, config.output_extension)


def predict_folder(
    config: PredictionConfig,
    folder: str,
    out_dir: str,
    extension: str = ".nii.gz",
    max_files: int = -1,
) -> None:
    image_files = find_image_files(folder, extension)
    if max_files >= 0:
        image_files = image_files[:max_files]
    if not image_files:
        raise FileNotFoundError(f"No images with extension {extension!r} found in {folder}")
    predict_images(config, image_files, out_dir, extension=extension)
