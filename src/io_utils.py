import os
import sys
import torch
import numpy as np

sys.path.append(os.getcwd())
import vtk_utils.vtk_utils as vtu


def read_image(img_fn):
    img = vtu.load_vtk_image(img_fn)
    x, y, z = img.GetDimensions()
    py_img = vtu.vtk_to_numpy(img.GetPointData().GetScalars()).reshape(z, y, x).transpose(2, 1, 0).astype(np.float32)
    img = torch.tensor(py_img)

    return img


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, output_fn, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.output_fn = output_fn

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                "model": model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, self.output_fn)

