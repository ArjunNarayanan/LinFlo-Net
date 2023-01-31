import torch
from src.dataset import ImageSegmentationDataset
import numpy as np
import vtk_utils.vtk_utils as vtu
import vtk

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def dice_score(mask1, mask2, epsilon=1e-7):
    num_intersect = (torch.logical_and(mask1, mask2)).count_nonzero()
    card1 = mask1.count_nonzero()
    card2 = mask2.count_nonzero()
    dice = 2 * num_intersect / (card1 + card2 + epsilon)
    return dice.item()


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


model_fn = "output/train_model/best_model_dict.pth"
model = torch.load(model_fn, map_location=device)["model"]
model.eval()

dataset_fn = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartDataSegmentation/validation"
dataset = ImageSegmentationDataset(dataset_fn)
data = dataset[8]

img = data["image"].unsqueeze(0)
with torch.no_grad():
    predicted_logits = model(img).squeeze(0)


gt_seg = data["segmentation"].squeeze(0)
scores = all_classes_dice_score(predicted_logits, gt_seg)

predicted_seg = predicted_logits.argmax(dim=0)
np_pred = predicted_seg.numpy().transpose(2, 1, 0).ravel()
vtk_seg_arr = vtu.numpy_to_vtk(np_pred)
vtk_im = vtk.vtkImageData()
vtk_im.GetPointData().SetScalars(vtk_seg_arr)
vtk_im.SetDimensions(3*[128])
vtk_im.SetSpacing(3*[1.0/128])
vtu.write_vtk_image(vtk_im, "test/TEV4P1CTAI.vti")

# seg_fn = "/Users/arjun/Documents/Research/SimCardio/Datasets/HeartDataSegmentation/validation/vtk_segmentation/CT_1.vti"
# vtk_seg = vtu.load_vtk_image(seg_fn)
