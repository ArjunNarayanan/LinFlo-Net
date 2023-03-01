import os
import sys

sys.path.append(os.getcwd())
from src.dataset import ImageSegmentationMeshDataset
import torch
from src.template import Template
from src.utilities import occupancy_map
from src.segment_flow import *


dataset_folder = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartDataSegmentation/validation"
dataset = ImageSegmentationMeshDataset(dataset_folder)

model_fn = "output/segment_flow/direct-1/best_model_dict.pth"
model_data = torch.load(model_fn, map_location=torch.device("cpu"))
model = model_data["model"]

template_fn = "data/template/highres_template.vtp"
template = Template.from_vtk(template_fn)
template_verts = template.verts_packed().unsqueeze(0)

image = dataset[0]["image"].unsqueeze(0)
lt_deformed_vertices = model.pretrained_linear_transform(image, template_verts)
occupancy = occupancy_map(lt_deformed_vertices, 128).unsqueeze(0)
encoder_input = model.get_encoder_input(image, occupancy)

encoding = model.encoder(encoder_input)
predicted_segmentation = model.segment_decoder(encoding)

flow_field = Decoder.forward(model.flow_decoder, encoding)
flow_norm = torch.norm(flow_field, dim=1, keepdim=True)