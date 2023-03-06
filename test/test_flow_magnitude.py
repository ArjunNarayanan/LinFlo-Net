import os
import sys

sys.path.append(os.getcwd())
from src.dataset import ImageSegmentationMeshDataset
import torch
from src.template import Template
from src.utilities import occupancy_map
from src.segment_flow import *


def get_all_maximum_flow_magnitudes(model, dataset, template_verts):
    flow_magnitudes = []
    for (idx, data) in enumerate(dataset):
        print("processing file : ", idx)
        image = data["image"].unsqueeze(0)
        lt_deformed_vertices = model.pretrained_linear_transform(image, template_verts)
        occupancy = occupancy_map(lt_deformed_vertices, 128).unsqueeze(0)
        encoder_input = model.get_pre_encoding(image, occupancy)

        with torch.no_grad():
            encoding = model.encoder(encoder_input)
            flow_field = model.flow_decoder(encoding)

        flow_norm = torch.norm(flow_field, dim=1)
        max_flow = flow_norm.max()
        flow_magnitudes.append(max_flow)

    return flow_magnitudes




dataset_folder = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartDataSegmentation/validation"
dataset = ImageSegmentationMeshDataset(dataset_folder)

model_fn = "output/segment_flow/direct-1/best_model_dict.pth"
model_data = torch.load(model_fn, map_location=torch.device("cpu"))
model = model_data["model"]

template_fn = "data/template/highres_template.vtp"
template = Template.from_vtk(template_fn)
template_verts = template.verts_packed().unsqueeze(0)

flow_magnitudes = get_all_maximum_flow_magnitudes(model, dataset, template_verts)