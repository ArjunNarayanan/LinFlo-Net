import torch
import torch.nn.functional as func
from torch.nn.modules.loss import _Loss
import numpy as np
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency


class SoftDiceLoss(_Loss):
    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, ground_truth):
        assert prediction.ndim == 5
        assert ground_truth.ndim == 4, "Ground truth ndim : " + str(ground_truth.ndim)
        assert ground_truth.shape[0] == prediction.shape[0]
        assert prediction.shape[2:] == ground_truth.shape[1:]
        assert ground_truth.dtype == torch.long

        batch_size = prediction.shape[0]
        num_classes = prediction.shape[1]

        prediction = prediction.softmax(dim=1)
        prediction = prediction.reshape(batch_size, num_classes, -1)

        ground_truth = ground_truth.reshape(batch_size, -1)
        ground_truth = func.one_hot(ground_truth, num_classes)
        ground_truth = ground_truth.permute(0, 2, 1)

        intersection = (prediction * ground_truth).sum(dim=(0, 2))
        cardinality = (prediction + ground_truth).sum(dim=(0, 2))

        score = (2 * intersection + self.eps) / (cardinality + self.eps)
        score = score.mean()

        return 1.0 - score


def dice_score(mask1, mask2, epsilon=1e-7):
    num_intersect = (torch.logical_and(mask1, mask2)).count_nonzero()
    card1 = mask1.count_nonzero()
    card2 = mask2.count_nonzero()
    dice = 2 * num_intersect / (card1 + card2 + epsilon)
    return dice


def all_classes_dice_score(predicted_logits, gt_classes):
    assert predicted_logits.ndim == 4
    num_classes = predicted_logits.shape[0]
    assert all(np.unique(gt_classes) == range(num_classes))
    predicted_seg = predicted_logits.argmax(dim=0)
    scores = []
    for label in range(num_classes):
        pred_mask = predicted_seg == label
        gt_mask = gt_classes == label
        scores.append(dice_score(pred_mask, gt_mask))
    return scores


def chamfer_loss_between_meshes(mesh1, mesh2, norm):
    assert norm == 1 or norm == 2

    x = mesh1.verts_padded()
    x_normals = mesh1.verts_normals_padded()
    x_lengths = mesh1.num_verts_per_mesh()

    y = mesh2.verts_padded()
    y_normals = mesh2.verts_normals_padded()
    y_lengths = mesh2.num_verts_per_mesh()

    chd, chn = chamfer_distance(x, y, x_lengths=x_lengths, y_lengths=y_lengths,
                                x_normals=x_normals, y_normals=y_normals, norm=norm)
    return chd, chn


def average_chamfer_distance_between_meshes(mesh_list1, mesh_list2, norm):
    num_meshes = len(mesh_list2)
    assert len(mesh_list1) == num_meshes

    loss_list = [chamfer_loss_between_meshes(mesh_list1[idx], mesh_list2[idx], norm) for idx in range(num_meshes)]
    chd = [l[0] for l in loss_list]
    chn = [l[1] for l in loss_list]

    avg_chd = sum(chd) / num_meshes
    avg_chn = sum(chn) / num_meshes
    return avg_chd, avg_chn


def average_mesh_edge_loss(mesh_list):
    edge_loss = [mesh_edge_loss(mesh) for mesh in mesh_list]
    return sum(edge_loss) / len(edge_loss)


def average_laplacian_smoothing_loss(mesh_list):
    laplacian_loss = [mesh_laplacian_smoothing(mesh) for mesh in mesh_list]
    return sum(laplacian_loss) / len(laplacian_loss)


def average_normal_consistency_loss(mesh_list):
    normal_consistency_loss = [mesh_normal_consistency(mesh) for mesh in mesh_list]
    return sum(normal_consistency_loss) / len(normal_consistency_loss)
