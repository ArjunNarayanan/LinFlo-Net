import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
from pytorch3d.structures import join_meshes_as_batch
from torch.utils.data import DataLoader


class ImageSegmentationDataset(Dataset):
    def __init__(self, root_dir):
        csv_file = os.path.join(root_dir, "index.csv")
        assert os.path.isfile(csv_file), "Could not find index file in " + root_dir

        self.index_file = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.index_file)

    def get_file_name(self, item):
        return self.index_file.iloc[item, 0]

    def __getitem__(self, item):
        filename = self.get_file_name(item)

        data_fn = os.path.join(self.root_dir, "data", filename + ".pkl")

        assert os.path.isfile(data_fn), "Did not find file " + data_fn
        data = pickle.load(open(data_fn, "rb"))

        return data


def collate_fn(data):
    assert all("image" in d for d in data)
    assert all("segmentation" in d for d in data)
    assert all("meshes" in d for d in data)
    
    assert len(data) > 0
    # Check that all images and segmentations have the same number of dimensions
    assert all((d["image"].ndim == 4 for d in data))
    assert all((d["segmentation"].ndim == 3 for d in data))

    meshes = [d["meshes"] for d in data]
    num_meshes = [len(m) for m in meshes]
    assert len(num_meshes) > 0
    num_components = num_meshes[0]
    # Check that all meshes have the same number of components
    assert all([n == num_components for n in num_meshes])

    img = torch.stack([d["image"] for d in data])
    seg = torch.stack([d["segmentation"] for d in data])

    meshes = [join_meshes_as_batch([m[i] for m in meshes]) for i in range(num_components)]
    return {"image": img, "meshes": meshes, "segmentation": seg}


class ImageSegmentationMeshDataset(Dataset):
    def __init__(self, root_dir):
        csv_file = os.path.join(root_dir, "index.csv")
        assert os.path.isfile(csv_file), "Could not find index file in " + root_dir

        self.index_file = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.index_file)

    def get_file_name(self, item):
        return self.index_file.iloc[item, 0]

    def __getitem__(self, item):
        filename = self.get_file_name(item)

        data_fn = os.path.join(self.root_dir, "data", filename + ".pkl")

        assert os.path.isfile(data_fn), "Did not find file " + data_fn
        data = pickle.load(open(data_fn, "rb"))

        return data


def image_segmentation_mesh_dataloader(root_dir, batch_size=1, shuffle=True):
    dataset = ImageSegmentationMeshDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
