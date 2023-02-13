import torch
import sys
import os
sys.path.append(os.getcwd())
from src.dataset import ImageSegmentationMeshDataset
from src.unet_segment import UnetSegment
from src.loss import SoftDiceLoss
from torch.nn import CrossEntropyLoss
import time
import yaml
from torch.utils.data import DataLoader


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def bytes2GB(bytes):
    return bytes/1.0e9

def print_memory_allocated(prefix):
    print("\n" + prefix)
    print("\t" + str(bytes2GB(torch.cuda.memory_allocated(device))) + " GB")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


config_fn = "config/train_model.yml"
with open(config_fn, "r") as config_file:
    config = yaml.safe_load(config_file)


net_config = config["model"]
net = UnetSegment.from_dict(net_config)
net.to(device)

dice_loss_evaluator = SoftDiceLoss()
cross_entropy_loss_evaluator = CrossEntropyLoss()

print("Num model parameters : ", count_parameters(net), "\n\n")
print_memory_allocated("After model load : ")

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-4)

data_fn = config["data"]["train_folder"]
batch_size = config["train"]["batch_size"]
dataset = ImageSegmentationMeshDataset(data_fn)
dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)


start = time.perf_counter()
data = next(iter(dataloader))
img = data["image"].to(device).to(memory_format=torch.channels_last_3d)
ground_truth = data["segmentation"].squeeze(1).to(device)
stop = time.perf_counter()
print_memory_allocated("After next iter dataloader :")
print("Time : ", stop - start)


start = time.perf_counter()
prediction = net(img)
stop = time.perf_counter()
print_memory_allocated("After forward pass :")
print("Time : ", stop - start)


start = time.perf_counter()
dice = dice_loss_evaluator(prediction, ground_truth)
stop = time.perf_counter()
print_memory_allocated("After dice loss :")
print("Time : ", stop - start)

start = time.perf_counter()
cross_entropy = cross_entropy_loss_evaluator(prediction, ground_truth)
stop = time.perf_counter()
print_memory_allocated("After cross entropy loss :")
print("Time : ", stop - start)


loss = dice + cross_entropy

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