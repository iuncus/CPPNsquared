import numpy as np
import torch


def get_normalised_coordinate_grid(image_shape):
    width = np.linspace(-1, 1, image_shape[0])
    height = np.linspace(-1, 1, image_shape[1])
    mgrid = np.stack(np.meshgrid(width, height), axis=-1)
    mgrid = np.reshape(mgrid, [-1, 2])
    return mgrid

def normalize(tensor, axis, model):

    if axis % 2 == 0:
        max_vals = len(list(model.children()))
        min_vals = 0
    else:
        max_vals, _ = torch.max(tensor, dim=0)
        min_vals, _ = torch.min(tensor, dim=0)

    normalized_tensor = 2 * (tensor - min_vals) / (max_vals - min_vals) - 1
    return normalized_tensor