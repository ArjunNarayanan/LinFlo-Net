from src.dataset import *
import time


def time_dataloader_pass():
    start = time.perf_counter()
    for sample in dataloader:
        image = sample["image"]
        print("shape : ", image.shape)
    stop = time.perf_counter()
    print("\n\nElapsed time : ", stop - start)


train_folder = "/global/scratch/users/arjunnarayanan/WholeHeartData/combined/ct/"
batch_size = 8
num_workers = 10

dataloader = image_segmentation_mesh_dataloader(
    train_folder,
    shuffle=True,
    batch_size=batch_size,
    num_workers=num_workers
)