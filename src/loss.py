import torch
import torch.nn.functional as func
from torch.nn.modules.loss import _Loss
import numpy as np


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
