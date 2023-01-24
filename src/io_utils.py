import os
import sys
import torch
import numpy as np

sys.path.append(os.getcwd())
import vtk_utils.vtk_utils as vtu


def read_image(img_fn):
    img = vtu.load_vtk_image(img_fn)
    x, y, z = img.GetDimensions()
    py_img = vtu.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
    img = torch.tensor(py_img)

    return img
