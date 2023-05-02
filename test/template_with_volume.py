from src.template import *
import pickle
import torch

device = torch.device("cpu")

template_fn = "data/template/template_with_volume.pkl"
template = TemplateWithVolume.from_pkl(template_fn)
template = template.to(device)

# batched_template = BatchTemplateWithVolume.from_single_template(template, 5)
