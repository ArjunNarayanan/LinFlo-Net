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


size = (128,128,128)
im_fn = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartDataSegmentation/raw_validation/ct/label/CT_1.nii.gz"
seg = sitk.ReadImage(im_fn)

arr = sitk.GetArrayFromImage(seg)

# transformed, reference = pre.resample_spacing(seg, template_size=size, order=0)
# re_transformed = pre.centering(transformed, seg, order=0)
# re_transformed_vtk, _ = vtu.exportSitk2VTK(re_transformed)
# vtu.write_vtk_image(re_transformed_vtk, "test/re_transformed_vtk.vti")
#
# seg_vtk, _ = vtu.exportSitk2VTK(seg)
# vtu.write_vtk_image(seg_vtk, "test/original.vti")