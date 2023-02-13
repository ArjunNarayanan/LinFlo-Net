import torch


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
