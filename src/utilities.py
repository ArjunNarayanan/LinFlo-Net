import torch
import numpy as np

def occupancy_map(vertices, num_grid_points):
    occupancy = torch.zeros(3 * [num_grid_points], device=vertices.device)
    idx = (vertices * num_grid_points).clamp(min=0, max=num_grid_points - 1).long()

    occupancy[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    occupancy = occupancy.unsqueeze(0)
    return occupancy


def verts_packed_of_batch(verts_list, batch):
    verts = [v[batch] for v in verts_list]
    verts = torch.cat(verts, dim=0)
    return verts


def batch_occupancy_map_from_vertices(vertices, batch_size, num_grid_points):
    verts_packed = [verts_packed_of_batch(vertices, b) for b in range(batch_size)]
    occ = [occupancy_map(v, num_grid_points) for v in verts_packed]
    occupancy = torch.stack(occ)
    return occupancy


def dice_score(pred, true):
    pred = pred.astype(int)
    true = true.astype(int)
    num_class = np.unique(true)

    # change to one hot
    dice_out = [None] * len(num_class)
    for i in range(1, len(num_class)):
        pred_c = pred == num_class[i]
        true_c = true == num_class[i]
        dice_out[i] = np.sum(pred_c * true_c) * 2.0 / (np.sum(pred_c) + np.sum(true_c))

    mask = (pred > 0) + (true > 0)
    dice_out[0] = np.sum((pred == true)[mask]) * 2. / (np.sum(pred > 0) + np.sum(true > 0))
    return dice_out
