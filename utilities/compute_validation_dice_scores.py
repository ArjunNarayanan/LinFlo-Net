import torch
import numpy as np
import argparse
import yaml
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from src.dataset import ImageSegmentationMeshDataset
from src.loss import all_classes_dice_score

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def get_config(config_fn):
    with open(config_fn, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def get_all_dice_scores(model, dataset, indices):
    scores = []

    for idx in indices:
        data = dataset[idx]
        fn = dataset.get_file_name(idx)
        print("PROCESSING FILE : " + fn)
        img = data["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_logits = model(img).squeeze(0)

        gt_seg = data["segmentation"].squeeze(0).to(device)
        scores.append(all_classes_dice_score(predicted_logits, gt_seg))

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model dice scores on validation dataset")
    parser.add_argument("-config", help="Path to configuration yaml file")
    args = parser.parse_args()

    config = get_config(args.config)

    dataset_fn = config["dataset"]
    dataset = ImageSegmentationMeshDataset(dataset_fn)

    model_fn = config["model"]
    model = torch.load(model_fn, map_location=device)["model"]
    model.to(device)
    model.eval()

    file_indices = range(len(dataset))
    scores = get_all_dice_scores(model, dataset, file_indices)
    scores = np.array(scores)

    dataset_filenames = [dataset.get_file_name(idx) for idx in file_indices]
    header = config["header"]
    assert len(header) == scores.shape[1]

    df = pd.DataFrame(scores, columns=header, index=dataset_filenames)

    out_folder = config["output_folder"]
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    filename = config["output_file"]
    out_fn = os.path.join(out_folder, filename)
    df.to_csv(out_fn)
