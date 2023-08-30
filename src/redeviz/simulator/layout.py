from __future__ import annotations
import numpy as np
from numpy.random import uniform
import pandas as pd
from plotnine import *
from numpy.random import normal
from shapely.geometry import MultiPoint, Polygon
from shapely.affinity import rotate, translate
from shapely.prepared import prep
from shapely import wkt
from redeviz.simulator.utils import radian2angle, uniform2D
import logging

class Layout(object):
    def __init__(self, x_range: int, y_range: int, nucleis_radius: float) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.nucleis_radius = nucleis_radius
        self.cell_map = None
    
    def __len__(self) -> int:
        return len(self.cell_map.geoms)
    
    def simulate(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    @classmethod
    def get_simulate_info(cls, cfg: dict) -> dict:
        raise NotImplementedError()

    def translate(self, xoff: float, yoff: float) -> None:
        self.cell_map = translate(self.cell_map, xoff, yoff)

    def clip(self, polygon: Polygon) -> None:
        prepared_polygon = prep(polygon)
        self.cell_map = MultiPoint([point for point in self.cell_map.geoms if prepared_polygon.contains(point)])

    @classmethod
    def is_circle_overlap(cls, mu1: float, R1: float, mu2: float, R2: float) -> bool:
        dist_cutoff = R1 + R2
        diff_mu = mu1 - mu2
        dist = np.sqrt(np.dot(diff_mu, diff_mu))
        return dist < dist_cutoff

    @classmethod
    def rect_clip(cls, coords_arr: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
        coords_arr[coords_arr[:, 0] < xmin, 0] = xmin
        coords_arr[coords_arr[:, 0] > xmax, 0] = xmax
        coords_arr[coords_arr[:, 1] < ymin, 1] = ymin
        coords_arr[coords_arr[:, 1] > ymax, 1] = ymax
        return coords_arr

    @classmethod
    def update_coords(cls, coords_arr: np.ndarray, circle_radius: float, 
        xmin: float, xmax: float, ymin: float, ymax: float):
        coord_num = coords_arr.shape[0]
        x_coords = coords_arr[:, 0]
        y_coords = coords_arr[:, 1]
        x_diff_mat = x_coords[:, np.newaxis] - x_coords
        y_diff_mat = y_coords[:, np.newaxis] - y_coords
        cutoff = 4 * circle_radius ** 2
        dist_square_mat = x_diff_mat ** 2 + y_diff_mat ** 2 + np.diag([cutoff]*coord_num)
        force_mat = cutoff / (dist_square_mat)
        force_mat[dist_square_mat >= cutoff] = 0
        max_rest_dist = np.sqrt(cutoff) - np.sqrt(dist_square_mat.min())
        ave_rest_dist = np.sqrt(cutoff) - np.sqrt(dist_square_mat.mean())
        if force_mat.max() == 0:
            return coords_arr, max_rest_dist, ave_rest_dist
        weight_mat = force_mat * circle_radius / np.sqrt(dist_square_mat) / 10
        delta_x = (x_diff_mat * weight_mat).sum(0)
        delta_y = (y_diff_mat * weight_mat).sum(0)
        max_delta = circle_radius / 2
        min_delta = -1 * max_delta
        delta_x[delta_x > max_delta] = max_delta
        delta_x[delta_x < min_delta] = min_delta
        delta_y[delta_y > max_delta] = max_delta
        delta_y[delta_y < min_delta] = min_delta
        res = coords_arr - np.array([delta_x, delta_y]).transpose()
        res = cls.rect_clip(res, xmin, xmax, ymin, ymax)
        return res, max_rest_dist, ave_rest_dist

    def plot(self, fname: str, polygon=None) -> None:
        if self.cell_map is None:
            raise ValueError()
        x = [point.x for point in self.cell_map.geoms]
        y = [point.y for point in self.cell_map.geoms]
        point_df = pd.DataFrame({
            "x": x,
            "y": y
        })
        p = ggplot()
        if polygon is not None:
            polygon_x, polygon_y = polygon.boundary.xy
            polygon_df = pd.DataFrame({
            "x": polygon_x,
            "y": polygon_y
            })
            p = p + geom_path(data=polygon_df, mapping=aes(x="x", y="y"), color="blue", size=0.3)
        p = p + geom_point(data=point_df, mapping=aes(x="x", y="y"), size=2, color="black") + \
            theme_bw() + \
            labs(x="X", y="Y") + \
            lims(x=(0, self.x_range), y=(0, self.y_range)) + \
            theme(
                text = element_text(family="Arial", size=5),
                title = element_text(family="Arial", size=6),
                axis_text = element_text(color = "black"),
                legend_position = "none",
                panel_grid = element_blank()
            )
        ggsave(p, filename=fname, width=5, height=5, limitsize=False, units="cm")


class RandomLayout(Layout):
    def simulate(self, alpha: float, circle_radius: float, max_iter=100000) -> None:
        assert min(self.x_range, self.y_range) > (2 * circle_radius)
        circle_radius = max(circle_radius, 1.2 * self.nucleis_radius)
        bounds = circle_radius, self.x_range-circle_radius, circle_radius, self.y_range-circle_radius
        x_range = self.x_range - 2 * circle_radius
        y_range = self.y_range - 2 * circle_radius
        N = int(np.floor(0.95 * alpha * x_range * y_range / (4 * circle_radius ** 2)))
        max_rest_dist = None
        while True:
            centers = uniform2D(*bounds, N)
            for index in range(max_iter):
                centers = np.unique(centers, axis=0)
                centers, max_rest_dist, ave_rest_dist = self.update_coords(centers, circle_radius, *bounds)
                if max_rest_dist == 0:
                    break
            if max_rest_dist == 0:
                break
        self.cell_map = MultiPoint(centers)

    @classmethod
    def get_simulate_info(cls, cfg: dict) -> dict:
        alpha = 1
        circle_radius = 12000
        if "layout_alpha" in cfg.keys():
            alpha = cfg["layout_alpha"]
        if "layout_circle_radius" in cfg.keys():
            circle_radius = cfg["layout_circle_radius"]
        simu_info = {
            "alpha": alpha,
            "circle_radius": circle_radius
        }
        return simu_info


class RectangleLayout(Layout):
    def simulate(self, a: float, b: float, sigma_x: float, sigma_y: float, phi: float, max_iter=10000) -> None:
        R = np.sqrt(self.x_range**2+self.y_range**2)/2
        coords = np.array([(x, y) for x in np.arange(-1*R, R, a) for y in np.arange(-1*R, R, b)])
        coords[:, 0] = coords[:, 0] + normal(0, sigma_x, coords.shape[0])
        coords[:, 1] = coords[:, 1] + normal(0, sigma_y, coords.shape[0])
        radius = 1.2 * self.nucleis_radius
        for index in range(max_iter):
            coords = np.unique(coords, axis=0)
            coords, max_rest_dist, ave_rest_dist = self.update_coords(coords, radius, -1*R, R, -1*R, R)
            if max_rest_dist == 0:
                break
        points = MultiPoint(coords)
        points = translate(rotate(points, radian2angle(phi)), self.x_range/2, self.y_range/2)
        self.cell_map = MultiPoint([point for point in points.geoms if point.x > 0 and point.y > 0 and point.x < self.x_range and point.y < self.y_range])

    @classmethod
    def get_simulate_info(cls, cfg: dict) -> dict:
        a = 12000
        b = 12000
        sigma_x = a / 5
        sigma_y = a / 5
        phi = uniform(0, np.pi)
        if "layout_a" in cfg.keys():
            a = cfg["layout_a"]
        if "layout_b" in cfg.keys():
            b = cfg["layout_b"]
        if "layout_sigma_x" in cfg.keys():
            sigma_x = cfg["layout_sigma_x"]
        if "layout_sigma_y" in cfg.keys():
            sigma_y = cfg["layout_sigma_y"]
        if "layout_phi" in cfg.keys():
            phi = cfg["layout_phi"]
        simu_info = {
            "a": a,
            "b": b,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "phi": phi
        }
        return simu_info


class TriangleLayout(Layout):
    def simulate(self, a: float, sigma_x: float, sigma_y: float, phi: float, max_iter=10000) -> None:
        R = np.sqrt(self.x_range**2+self.y_range**2)/2
        coords = list()
        x0 = np.arange(-1*R, R, a)
        for y in np.arange(-1*R, R, np.sqrt(3)*a):
            coords += [(x, y) for x in x0]
            coords += [(x+a/2, y+np.sqrt(3)*a/2) for x in x0]
        coords = np.array(coords)
        coords[:, 0] = coords[:, 0] + normal(0, sigma_x, coords.shape[0])
        coords[:, 1] = coords[:, 1] + normal(0, sigma_y, coords.shape[0])
        radius = 1.2 * self.nucleis_radius
        for index in range(max_iter):
            coords = np.unique(coords, axis=0)
            coords, max_rest_dist, ave_rest_dist = self.update_coords(coords, radius, -1*R, R, -1*R, R)
            if max_rest_dist == 0:
                break
        points = MultiPoint(coords)
        points = translate(rotate(points, radian2angle(phi)), self.x_range/2, self.y_range/2)
        self.cell_map = MultiPoint([point for point in points.geoms if point.x > 0 and point.y > 0 and point.x < self.x_range and point.y < self.y_range])

    @classmethod
    def get_simulate_info(cls, cfg: dict) -> dict:
        a = 12000
        sigma_x = a / 5
        sigma_y = a / 5
        phi = uniform(0, np.pi)
        if "layout_a" in cfg.keys():
            a = cfg["layout_a"]
        if "layout_sigma_x" in cfg.keys():
            sigma_x = cfg["layout_sigma_x"]
        if "layout_sigma_y" in cfg.keys():
            sigma_y = cfg["layout_sigma_y"]
        if "layout_phi" in cfg.keys():
            phi = cfg["layout_phi"]
        simu_info = {
            "a": a,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "phi": phi
        }
        return simu_info


class HexagonalLayout(Layout):
    def simulate(self, a: float, sigma_x: float, sigma_y: float, phi: float, max_iter=10000) -> None:
        R = np.sqrt(self.x_range**2+self.y_range**2)/2
        coords = list()
        x0 = np.arange(-1*R, R, 3*a)
        for y in np.arange(-1*R, R, np.sqrt(3)*a):
            coords += [(x, y) for x in x0]
            coords += [(x+a, y) for x in x0]
            coords += [(x-a/2, y+np.sqrt(3)*a/2) for x in x0]
            coords += [(x+1.5*a, y+np.sqrt(3)*a/2) for x in x0]
        coords = np.array(coords)
        coords[:, 0] = coords[:, 0] + normal(0, sigma_x, coords.shape[0])
        coords[:, 1] = coords[:, 1] + normal(0, sigma_y, coords.shape[0])
        radius = 1.2 * self.nucleis_radius
        for index in range(max_iter):
            coords = np.unique(coords, axis=0)
            coords, max_rest_dist, ave_rest_dist = self.update_coords(coords, radius, -1*R, R, -1*R, R)
            if max_rest_dist == 0:
                break
        points = MultiPoint(coords)
        points = translate(rotate(points, radian2angle(phi)), self.x_range/2, self.y_range/2)
        self.cell_map = MultiPoint([point for point in points.geoms if point.x > 0 and point.y > 0 and point.x < self.x_range and point.y < self.y_range])

    @classmethod
    def get_simulate_info(cls, cfg: dict) -> dict:
        a = 12000
        sigma_x = a / 5
        sigma_y = a / 5
        phi = uniform(0, np.pi)
        if "layout_a" in cfg.keys():
            a = cfg["layout_a"]
        if "layout_sigma_x" in cfg.keys():
            sigma_x = cfg["layout_sigma_x"]
        if "layout_sigma_y" in cfg.keys():
            sigma_y = cfg["layout_sigma_y"]
        if "layout_phi" in cfg.keys():
            phi = cfg["layout_phi"]
        simu_info = {
            "a": a,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "phi": phi
        }
        return simu_info


class ParallelogramLayout(Layout):
    def simulate(self, a: float, b: float, alpha: float, sigma_x: float, sigma_y: float, phi: float, max_iter=10000) -> None:
        R = np.sqrt(self.x_range**2+self.y_range**2)/2
        coords = np.array([(x+((y*np.cos(alpha))%a), y) for x in np.arange(-1*R, R, a) for y in np.arange(-1*R, R, b*np.sin(alpha))])
        coords[:, 0] = coords[:, 0] + normal(0, sigma_x, coords.shape[0])
        coords[:, 1] = coords[:, 1] + normal(0, sigma_y, coords.shape[0])
        radius = 1.2 * self.nucleis_radius
        for index in range(max_iter):
            coords = np.unique(coords, axis=0)
            coords, max_rest_dist, ave_rest_dist = self.update_coords(coords, radius, -1*R, R, -1*R, R)
            if max_rest_dist == 0:
                break
        points = MultiPoint(coords)
        points = translate(rotate(points, radian2angle(phi)), self.x_range/2, self.y_range/2)
        self.cell_map = MultiPoint([point for point in points.geoms if point.x > 0 and point.y > 0 and point.x < self.x_range and point.y < self.y_range])

    @classmethod
    def get_simulate_info(cls, cfg: dict) -> dict:
        a = 12000
        b = 12000
        sigma_x = a / 5
        sigma_y = a / 5
        alpha = uniform(0, np.pi)
        phi = uniform(0, np.pi)
        if "layout_a" in cfg.keys():
            a = cfg["layout_a"]
        if "layout_b" in cfg.keys():
            b = cfg["layout_b"]
        if "layout_sigma_x" in cfg.keys():
            sigma_x = cfg["layout_sigma_x"]
        if "layout_sigma_y" in cfg.keys():
            sigma_y = cfg["layout_sigma_y"]
        if "layout_alpha" in cfg.keys():
            alpha = cfg["layout_alpha"]
        if "layout_phi" in cfg.keys():
            phi = cfg["layout_phi"]
        simu_info = {
            "a": a,
            "b": b,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "alpha": alpha,
            "phi": phi
        }
        return simu_info


class CustomLayout(Layout):
    def simulate(self, cell_map: MultiPoint) -> None:
        self.cell_map = cell_map

    @classmethod
    def get_simulate_info(cls, cfg: dict) -> dict:
        assert "layout_cell_map" in cfg.keys()
        cell_map = wkt.loads(cfg["layout_cell_map"])
        assert cell_map.type == "MultiPoint"
        simu_info = {
            "cell_map": cell_map
        }
        return simu_info
