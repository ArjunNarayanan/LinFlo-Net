import torch
import sys
import os
sys.path.append(os.getcwd())
from src.dataset import image_segmentation_mesh_dataloader
from src.linear_transform import LinearTransformWithEncoder, LinearTransformNet
from src.loss import SoftDiceLoss, average_chamfer_distance_between_meshes
from src.template import Template, BatchTemplate
from torch.nn import CrossEntropyLoss
import time
import yaml


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def bytes2GB(bytes):
    return bytes/1.0e9

def print_memory_allocated(prefix):
    print("\n" + prefix)
    print("\t" + str(bytes2GB(torch.cuda.memory_allocated(device))) + " GB")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


config_fn = "config/linear_transform/direct_linear_transform/config.yml"
with open(config_fn, "r") as config_file:
    config = yaml.safe_load(config_file)


net_config = config["model"]
net = LinearTransformNet.from_dict(net_config)
net.to(device)

print("Num model parameters : ", count_parameters(net), "\n\n")
print_memory_allocated("After model load : ")

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-4)

data_fn = config["data"]["train_folder"]
batch_size = config["train"]["batch_size"]
dataloader = image_segmentation_mesh_dataloader(data_fn, shuffle=True, batch_size=batch_size)

template_fn = config["data"]["template_filename"]
template = Template.from_vtk(template_fn, device=device)

batched_template = BatchTemplate.from_single_template(template, batch_size)
batched_verts = batched_template.batch_vertex_coordinates()

start = time.perf_counter()
data = next(iter(dataloader))
img = data["image"].to(device).to(memory_format=torch.channels_last_3d)
gt_meshes = [m.to(device) for m in data["meshes"]]
stop = time.perf_counter()
print_memory_allocated("After next iter dataloader :")
print("Time : ", stop - start)


start = time.perf_counter()
deformed_verts = net(img, batched_verts)
stop = time.perf_counter()
print_memory_allocated("After forward pass :")
print("Time : ", stop - start)

batched_template.update_batched_vertices(deformed_verts, detach=False)

start = time.perf_counter()
chd, _ = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, 1)
stop = time.perf_counter()
print_memory_allocated("After chamfer loss :")
print("Time : ", stop - start)

loss = chd

start = time.perf_counter()
loss.backward()
stop = time.perf_counter()
print_memory_allocated("After backward pass :")
print("Time : ", stop - start)


start = time.perf_counter()
optimizer.step()
stop = time.perf_counter()
print_memory_allocated("After optimizer step :")
print("Time : ", stop - start)