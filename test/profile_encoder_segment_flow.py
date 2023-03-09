import os
import torch.optim.lr_scheduler
from torch.nn import CrossEntropyLoss
import sys

sys.path.append(os.getcwd())
from src.segment_flow import *
from src.utilities import batch_occupancy_map_from_vertices
from src.integrator import IntegrateFlowDivRK4
from src.loss import *
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


def initialize_model(model_config):
    pretrained_encoder = torch.load(model_config["pretrained_encoder"], map_location=device)
    pretrained_linear_transform = torch.load(model_config["pretrained_linear_transform"], map_location=device)
    encoder = Unet.from_dict(model_config["encoder"])

    decoder_input_channels = model_config["encoder"]["uparm_channels"][-1]
    decoder_hidden_channels = model_config["segment"]["decoder_hidden_channels"]
    decoder_output_channels = model_config["segment"]["output_channels"]
    segment_decoder = Decoder(decoder_input_channels, decoder_hidden_channels, decoder_output_channels)

    # since we add occupancy as a new channel, input channels increases by one
    decoder_input_channels = decoder_input_channels + 1
    decoder_hidden_channels = model_config["flow"]["decoder_hidden_channels"]
    flow_clip_value = model_config["flow"]["clip"]
    flow_decoder = FlowDecoder(decoder_input_channels, decoder_hidden_channels, flow_clip_value)

    integrator = IntegrateFlowDivRK4(model_config["integrator"]["num_steps"])
    input_shape = model_config["encoder"]["input_shape"]
    net = EncodeLinearTransformSegmentFlow(input_shape,
                                           pretrained_encoder,
                                           pretrained_linear_transform,
                                           encoder,
                                           segment_decoder,
                                           flow_decoder,
                                           integrator)
    return net


config_fn = "config/segment_flow/model-7/config.yml"
with open(config_fn, "r") as config_file:
    config = yaml.safe_load(config_file)

net_config = config["model"]
input_shape = net_config["encoder"]["input_shape"]
net = initialize_model(net_config)
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

flow_div = FlowDiv(input_shape)
cross_entropy_evaluator = CrossEntropyLoss(reduction="mean")

start = time.perf_counter()
data = next(iter(dataloader))
img = data["image"].to(device).to(memory_format=torch.channels_last_3d)
gt_meshes = [m.to(device) for m in data["meshes"]]
gt_segmentation = data["segmentation"].to(device)
stop = time.perf_counter()
print_memory_allocated("After next iter dataloader :")
print("Time : ", stop - start)

start = time.perf_counter()
with torch.no_grad():
    pre_encoding = net.get_encoder_input(img)
stop = time.perf_counter()
print_memory_allocated("After pre-encoder :")
print("Time : ", stop - start)

start = time.perf_counter()
with torch.no_grad():
    lt_deformed_vertices = net.pretrained_linear_transform(pre_encoding, batched_verts)
stop = time.perf_counter()
print_memory_allocated("After linear transform :")
print("Time : ", stop - start)


start = time.perf_counter()
encoding = net.encoder(pre_encoding)
stop = time.perf_counter()
print_memory_allocated("After encoder :")
print("Time : ", stop - start)


start = time.perf_counter()
segmentation = net.segment_decoder(encoding)
stop = time.perf_counter()
print_memory_allocated("After segmentation :")
print("Time : ", stop - start)


start = time.perf_counter()
encoding = net.get_flow_decoder_input(encoding, lt_deformed_vertices, batch_size, net.input_size)
stop = time.perf_counter()
print_memory_allocated("After occupancy map :")
print("Time : ", stop - start)

start = time.perf_counter()
flow = net.flow_decoder(encoding)
flow_and_div = net.flow_div.get_flow_div(flow)
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
cross_entropy_loss = cross_entropy_evaluator(segmentation, gt_segmentation)
stop = time.perf_counter()
print_memory_allocated("After cross entropy loss :")
print("Time : ", stop - start)


start = time.perf_counter()
chd, chn = average_chamfer_distance_between_meshes(batched_template.meshes_list, gt_meshes, 1)
stop = time.perf_counter()
print_memory_allocated("After chamfer loss :")
print("Time : ", stop - start)

start = time.perf_counter()
divergence_loss = average_divergence_loss(div_integral)
stop = time.perf_counter()
print_memory_allocated("After divergence loss :")
print("Time : ", stop - start)

start = time.perf_counter()
edge_loss = average_mesh_edge_loss(batched_template.meshes_list)
laplace_loss = average_laplacian_smoothing_loss(batched_template.meshes_list)
normal_loss = average_normal_consistency_loss(batched_template.meshes_list)
stop = time.perf_counter()
print_memory_allocated("After geometric losses :")
print("Time : ", stop - start)


loss = chd + chn + divergence_loss + cross_entropy_loss + edge_loss + laplace_loss + normal_loss

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
