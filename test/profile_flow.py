import os
import torch.optim.lr_scheduler
import sys

sys.path.append(os.getcwd())
from src.flow import EncodeLinearTransformFlow, Flow, FlowDiv
from src.utilities import batch_occupancy_map_from_vertices
from src.integrator import IntegrateFlowDivRK4
from src.loss import average_chamfer_distance_between_meshes
from src.dataset import image_segmentation_mesh_dataloader
from src.template import Template, BatchTemplate
import yaml
import time

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def bytes2GB(bytes):
    return bytes / 1.0e9


def print_memory_allocated(prefix):
    print("\n" + prefix)
    print("\t" + str(bytes2GB(torch.cuda.memory_allocated(device))) + " GB")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mean_divergence_loss(div_integral):
    mean_divergence = [(-fd).exp().mean() for fd in div_integral]
    mean_divergence = sum(mean_divergence) / len(mean_divergence)
    return mean_divergence


config_fn = "output/linear_transform/config.yml"
with open(config_fn, "r") as config_file:
    config = yaml.safe_load(config_file)

net_config = config["model"]
pretrained_encoder = torch.load(net_config["pretrained_encoder"], map_location=device)
linear_transform = torch.load(net_config["pretrained_linear_transform"], map_location=device)
flow = Flow.from_dict(config["flow"])
integrator = IntegrateFlowDivRK4(config["integrator"]["num_steps"])
net = EncodeLinearTransformFlow(pretrained_encoder, linear_transform, flow, integrator)
net.to(device)

print("Num model parameters : ", count_parameters(net), "\n\n")
print_memory_allocated("After model load : ")

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

data_fn = config["data"]["train_folder"]
batch_size = config["train"]["batch_size"]
dataloader = image_segmentation_mesh_dataloader(data_fn, shuffle=True, batch_size=batch_size)

template_fn = config["data"]["template_filename"]
template = Template.from_vtk(template_fn, device=device)

batched_template = BatchTemplate.from_single_template(template, batch_size)
batched_verts = batched_template.batch_vertex_coordinates()

flow_div = FlowDiv(net.flow.input_shape)

start = time.perf_counter()
data = next(iter(dataloader))
img = data["image"].to(device).to(memory_format=torch.channels_last_3d)
gt_meshes = [m.to(device) for m in data["meshes"]]
stop = time.perf_counter()
print_memory_allocated("After next iter dataloader :")
print("Time : ", stop - start)

start = time.perf_counter()
encoding = net.encoder(img)
encoding = torch.cat([img, encoding], dim=1)
stop = time.perf_counter()
print_memory_allocated("After encoder :")
print("Time : ", stop - start)

start = time.perf_counter()
lt_deformed_vertices = net.linear_transform(encoding, batched_verts)
stop = time.perf_counter()
print_memory_allocated("After linear transform :")
print("Time : ", stop - start)

start = time.perf_counter()
occupancy = batch_occupancy_map_from_vertices(lt_deformed_vertices, batch_size, net.flow.input_shape)
encoding = torch.cat([encoding, occupancy], dim=1)
stop = time.perf_counter()
print_memory_allocated("After occupancy map :")
print("Time : ", stop - start)

start = time.perf_counter()
flow = net.flow.get_flow_field(encoding)
flow_and_div = flow_div.get_flow_div(flow)
stop = time.perf_counter()
print_memory_allocated("After generating flow and div :")
print("Time : ", stop - start)

start = time.perf_counter()
deformed_verts, div_integral = net.integrator.integrate_flow_and_div(flow_and_div, lt_deformed_vertices)
stop = time.perf_counter()
print_memory_allocated("After integrating flow and div :")
print("Time : ", stop - start)

start = time.perf_counter()
batched_template.update_batched_vertices(deformed_verts, detach=False)
stop = time.perf_counter()
print_memory_allocated("After update batched_vertices :")
print("Time : ", stop - start)

start = time.perf_counter()
chd, chn = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, 1)
divergence_loss = mean_divergence_loss(div_integral)
stop = time.perf_counter()
print_memory_allocated("After loss :")
print("Time : ", stop - start)

loss = chd + chn + divergence_loss

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
