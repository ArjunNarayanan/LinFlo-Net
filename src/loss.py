import torch
import torch.nn.functional as func
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):
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
        prediction = prediction.view(batch_size, num_classes, -1)

        ground_truth = ground_truth.view(batch_size, -1)
        ground_truth = func.one_hot(ground_truth, num_classes)
        ground_truth = ground_truth.permute(0, 2, 1)

        intersection = (prediction * ground_truth).sum(dim=(0, 2))
        cardinality = (prediction + ground_truth).sum(dim=(0, 2))

        score = (2*intersection + self.eps)/(cardinality + self.eps)
        score = score.mean()

        return 1.0 - score
