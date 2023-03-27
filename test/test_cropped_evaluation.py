import os
import sys
sys.path.append(os.getcwd())
from utilities.predict_test_meshes import *


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

model_fn = "/global/scratch/users/arjunnarayanan/HeartDataSegmentation/trained_models/segment_flow/model-3/best_model_dict.pth"
model_data = torch.load(model_fn, map_location=device)
model = model_data["model"]

template_fn = "data/template/highres_template.vtp"
template = Template.from_vtk(template_fn)

image_fn = "output/test/mr_test_2031_image_cropped.nii.gz"
info = {"input_size": 128}

output_dir = "output/test/"
prediction = Prediction(info, model, template, output_dir, "mr")

prediction.set_image_info(image_fn)
prediction.predict_mesh()

out_fn = os.path.join(output_dir, "mr_test_2031_image_cropped.vtp")
vtu.write_vtk_polydata(prediction.prediction, out_fn)

prediction.mesh_to_segmentation()
seg_fn = os.path.join(output_dir, "mr_test_2031_image_cropped_pred.nii.gz")
ref_im, M = vtu.exportSitk2VTK(prediction.original_image)
print("Writing nifti with name: ", seg_fn)
vtu.vtk_write_mask_as_nifty(prediction.segmentation, M, prediction.image_fn, seg_fn)