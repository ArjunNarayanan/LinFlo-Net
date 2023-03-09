import os
import torch.optim.lr_scheduler
from torch.nn import CrossEntropyLoss
import sys

sys.path.append(os.getcwd())
from src.segment_flow import *
from src.utilities import batch_occupancy_map_from_vertices
from src.integrator import IntegrateFlowDivRK4
from src.flow_loss import *
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
    decoder_hidden_channels = model_config["decoder"]["hidden_channels"]
    decoder_output_channels = model_config["decoder"]["output_channels"]
    unified_decoder = UnifiedDecoder(decoder_input_channels, decoder_hidden_channels, decoder_output_channels)

    flow_clip_value = model_config["clip_flow"]
    integrator = IntegrateFlowDivRK4(model_config["integrator"]["num_steps"])
    input_shape = model_config["encoder"]["input_shape"]
    net = UnifiedSegmentFlow(input_shape,
                            pretrained_encoder,
                            pretrained_linear_transform,
                            encoder,
                            unified_decoder,
                            integrator,
                            flow_clip_value)
    return net


config_fn = "config/segment_flow/unified-1/config.yml"
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
soft_dice_evaluator = SoftDiceLoss()
loss_evaluators = {"cross_entropy": cross_entropy_evaluator,
                    "dice": soft_dice_evaluator}
loss_config = config["loss"]

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
    pre_encoding = net.get_pre_encoding(img)
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
encoding = net.get_encoding(pre_encoding, lt_deformed_vertices)
stop = time.perf_counter()
print_memory_allocated("After encoder :")
print("Time : ", stop - start)


start = time.perf_counter()
decoding = net.unified_decoder(encoding)
stop = time.perf_counter()
print_memory_allocated("After decoding :")
print("Time : ", stop - start)

start = time.perf_counter()
flow_field = net.get_flow_from_decoding(decoding)
flow_and_div = net.flow_div.get_flow_div(flow_field)
stop = time.perf_counter()
print_memory_allocated("After get flow from decoder :")
print("Time : ", stop - start)

start = time.perf_counter()
predicted_segmentation = net.get_segmentation_from_decoding(decoding)
stop = time.perf_counter()
print_memory_allocated("After get segmentation from decoder :")
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

predictions = {"meshes": batched_template.meshes_list,
               "segmentation": predicted_segmentation,
               "divergence_integral": div_integral}
ground_truth = {"meshes": gt_meshes,
                "segmentation": gt_segmentation}
start = time.perf_counter()
loss_components = compute_loss_components(predictions, ground_truth, loss_evaluators, loss_config)
stop = time.perf_counter()
print_memory_allocated("After computing loss :")
print("Time : ", stop - start)


loss = loss_components["total"]

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
