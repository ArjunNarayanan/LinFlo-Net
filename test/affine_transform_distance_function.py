import SimpleITK as sitk
from src.template import Template
from src.linear_transform import LinearTransformer
import numpy as np


translation_x = 0.1
translation_y = 0.
translation_z = 0.

img_fn = "test/output/distance.vtk"
img = sitk.ReadImage(img_fn)


affine_transform = sitk.AffineTransform(3)  # Create a 3D affine transformation
affine_transform.Translate([translation_x,translation_y,translation_z])

# Set the center of rotation (usually the center of the image)
affine_transform.SetCenter(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))

transformed_image = sitk.Resample(img, affine_transform)

out_fn = "test/output/translated_distance.vtk"
sitk.WriteImage(transformed_image, out_fn)