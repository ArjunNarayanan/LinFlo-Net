from src.template import *
import pickle
import torch

device = torch.device("cpu")

batch_size = 1
model_fn = "output/WholeHeartData/trained_models/mr/flow/model-1/best_model.pth"
data = torch.load(model_fn, map_location=device)
model = data["model"]

template_fn = "data/template/highres_template.vtp"
template = Template.from_vtk(template_fn)
template = template.to(device)

batched_template = BatchTemplate.from_single_template(template, batch_size)
batched_verts = batched_template.batch_vertex_coordinates()

image = torch.rand([batch_size, 1, 128, 128, 128])

predictions = model.predict(image, batched_verts)

lv_interior_fn = "data/template/lv_interior_torch.pkl"
lv_interior = pickle.load(open(lv_interior_fn, "rb"))