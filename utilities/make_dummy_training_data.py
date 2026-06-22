"""Generate a tiny synthetic dataset to smoke-test the training pipeline.

This fabricates the minimal data format expected by
``src.dataset.ImageSegmentationMeshDataset``:

    <root>/index.csv               # first column = sample name (no extension)
    <root>/data/<name>.pkl         # dict with image / segmentation / meshes

The ground-truth meshes are taken directly from the real template so that the
number of mesh components matches what the model produces. Images and
segmentations are random noise -- this only checks that the training loop runs
(forward, loss, backward, checkpoint), not that anything is learned.
"""
import os
import sys
import pickle
import argparse

import torch
import pandas as pd

sys.path.append(os.getcwd())
from src.template import Template
from pytorch3d.structures import Meshes


def write_split(root_dir, template, num_samples, input_shape):
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # A plain Meshes holding the template's components (len == num_components).
    gt_meshes = Meshes(verts=template.verts_list(), faces=template.faces_list())

    names = []
    for i in range(num_samples):
        name = f"sample_{i:03d}"
        sample = {
            "image": torch.rand(1, input_shape, input_shape, input_shape, dtype=torch.float32),
            "segmentation": torch.zeros(input_shape, input_shape, input_shape, dtype=torch.long),
            "meshes": gt_meshes,
        }
        with open(os.path.join(data_dir, name + ".pkl"), "wb") as f:
            pickle.dump(sample, f)
        names.append(name)

    pd.DataFrame({"file_name": names}).to_csv(os.path.join(root_dir, "index.csv"), index=False)
    print(f"Wrote {num_samples} samples to {root_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset for a training smoke test")
    parser.add_argument("-o", "--output", default="test/smoke_train_data", help="output root folder")
    parser.add_argument("-t", "--template", default="data/template/highres_template.vtp", help="template .vtp")
    parser.add_argument("--input_shape", type=int, default=16, help="cube side length of synthetic images")
    parser.add_argument("--num_train", type=int, default=2)
    parser.add_argument("--num_val", type=int, default=1)
    args = parser.parse_args()

    template = Template.from_vtk(args.template)
    print("Template components:", len(template.verts_list()))

    write_split(os.path.join(args.output, "train"), template, args.num_train, args.input_shape)
    write_split(os.path.join(args.output, "val"), template, args.num_val, args.input_shape)


if __name__ == "__main__":
    main()
