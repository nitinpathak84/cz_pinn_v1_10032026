import numpy as np


def make_regular_grid(xmin, xmax, ymin, ymax, nx=201, ny=301):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    return {
        "x": xx.reshape(-1, 1),
        "y": yy.reshape(-1, 1),
    }