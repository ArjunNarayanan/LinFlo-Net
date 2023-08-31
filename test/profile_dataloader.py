import os
import sys
sys.path.append(os.getcwd())
from src.dataset import *
import time

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def time_dataloader_pass(num_workers):
    dataloader = image_segmentation_mesh_dataloader(
        train_folder,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )

    start = time.perf_counter()
    for sample in dataloader:
        image = sample["image"].to(device)
        mesh = [m.to(device) for m in sample["meshes"]]
        print("shape : ", image.shape)
    stop = time.perf_counter()
    print("\n\nElapsed time : ", stop - start)


train_folder = "/global/scratch/users/arjunnarayanan/WholeHeartData/validation/ct/augment20/processed/"
batch_size = 8
num_workers = 2