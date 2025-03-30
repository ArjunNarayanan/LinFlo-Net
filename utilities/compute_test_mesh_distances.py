import os

import pandas as pd

from src.template import Template
from scipy.spatial.distance import directed_hausdorff
import glob


def compute_distance(mesh1, mesh2, num_components=5):
    assert len(mesh1) == num_components
    assert len(mesh2) == num_components
    distances = [directed_hausdorff(mesh1[idx], mesh2[idx]) for idx in range(num_components)]

    return distances


def compute_all_distances(gt_files, pred_files):
    distances = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        print("Processing : ", os.path.basename(gt_file))
        gt_mesh = Template.from_vtk(gt_file)
        pred_mesh = Template.from_vtk(pred_file)
        distances.append(compute_distance(gt_mesh, pred_mesh))


if __name__ == "__main__":
    gt_folder = "/Users/arjunnarayanan/Documents/Research/Simcardio/Datasets/Cardiovascular/ct/meshes"
    pred_folder = "/Users/arjunnarayanan/Documents/Research/Simcardio/Cardiovascular/cardiac-model-results/meshes"
    output_folder = "/Users/arjunnarayanan/Documents/Research/Simcardio/Cardiovascular/cardiac-model-results/distances"
    outputfile = "hausdorff_distance.csv"
    num_mesh_components = 5 # might only need first 5 meshes for cardiac components

    extension = ".vtp"

    gt_files = glob.glob(os.path.join(gt_folder, "*" + extension))
    filenames = [os.path.basename(gt) for gt in gt_files]
    pred_files = [os.path.join(pred_folder, filename) for filename in filenames]
    assert all(os.path.isfile(f) for f in pred_files)

    hausdorff_distances = compute_all_distances(gt_files, pred_files)
    data = pd.DataFrame(hausdorff_distances, index=filenames)

    output_file_path = os.path.join(output_folder, outputfile)
    data.to_csv(output_file_path)