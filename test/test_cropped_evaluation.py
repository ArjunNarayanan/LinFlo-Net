from utilities.predict_test_meshes import *
import os



model_fn = "output/segment_flow/model-3/best_model_dict.pth"
model_data = torch.load(model_fn, map_location=torch.device("cpu"))
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
