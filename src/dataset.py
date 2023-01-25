import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
import pickle


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