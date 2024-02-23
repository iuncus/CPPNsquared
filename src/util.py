import numpy as np


def get_normalised_coordinate_grid(image_shape):
    width = np.linspace(-1, 1, image_shape[0])
    height = np.linspace(-1, 1, image_shape[1])
    mgrid = np.stack(np.meshgrid(width, height), axis=-1)
    mgrid = np.reshape(mgrid, [-1, 2])
    return mgrid