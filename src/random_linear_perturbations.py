from src.linear_transform import LinearTransformer
import torch

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def get_random_scaling(batch_size, magnitude):
    perturbation = magnitude * (2 * torch.rand([batch_size, 3]) - 1)
    perturbation = 1 - perturbation
    perturbation = perturbation.to(device)
    return perturbation


def get_random_translation(batch_size, magnitude):
    perturbation = magnitude * (2 * torch.rand([batch_size, 3]) - 1)
    perturbation = perturbation.to(device)
    return perturbation


def get_random_rotation(batch_size, magnitude):
    perturbation = 2 * torch.pi * magnitude * (2 * torch.rand([batch_size, 3]) - 1)
    perturbation = perturbation.to(device)
    return perturbation


def get_random_linear_transformer(batch_size, scale_magnitude, translate_magnitude, rotate_magnitude):
    scale_params = get_random_scaling(batch_size, scale_magnitude)
    translate_params = get_random_translation(batch_size, translate_magnitude)
    rotate_params = get_random_rotation(batch_size, rotate_magnitude)
    return LinearTransformer(scale_params, translate_params, rotate_params)
