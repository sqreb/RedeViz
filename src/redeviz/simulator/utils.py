import numpy as np
from numpy.random import uniform, normal
from typing import Iterable

def uniform2D(xmin: float, xmax: float, ymin: float, ymax: float, N: int) -> np.ndarray:
    x_li = uniform(xmin, xmax, N)
    y_li = uniform(ymin, ymax, N)
    return np.array([x_li, y_li]).T

def iter_uniform2D(xmin: float, xmax: float, ymin: float, ymax: float, buffer_size=10000) -> Iterable[np.ndarray]:
    while True:
        x_li = uniform(xmin, xmax, buffer_size)
        y_li = uniform(ymin, ymax, buffer_size)
        for (x, y) in zip(x_li, y_li):
            yield np.array([x, y])

def iter_uniform(xmin: float, xmax: float, buffer_size=10000) -> Iterable[float]:
    while True:
        x_li = uniform(xmin, xmax, buffer_size)
        for x in x_li:
            yield x

def normal2D(mu_x: float, mu_y: float, sd_x: float, sd_y: float, phi: float, N: int) -> np.ndarray:
    x_li = normal(0, sd_x, N)
    y_li = normal(0, sd_y, N)
    coords = np.array([x_li, y_li]).T
    if phi != 0:
        M = np.array([[np.cos(phi), -1*np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        coords = np.dot(coords, M)
    coords[:, 0] = coords[:, 0] + mu_x
    coords[:, 1] = coords[:, 1] + mu_y
    return coords

def angle2radian(algle: float) -> float:
    return algle / 180 * np.pi

def radian2angle(radian: float) -> float:
    return radian / np.pi * 180

def compute_pairwise_dist(x_coords: np.ndarray, y_coords: np.ndarray, x2_coords=None, y2_coords=None, return_square=False) -> np.ndarray:
    if x2_coords is None:
        assert y2_coords is None
        x_diff_mat = x_coords[:, np.newaxis] - x_coords
        y_diff_mat = y_coords[:, np.newaxis] - y_coords
    else:
        assert y2_coords is not None
        x_diff_mat = x_coords[:, np.newaxis] - x2_coords
        y_diff_mat = y_coords[:, np.newaxis] - y2_coords
    dist_square_mat = x_diff_mat ** 2 + y_diff_mat ** 2
    if return_square:
        return dist_square_mat
    else:
        return np.sqrt(dist_square_mat)

