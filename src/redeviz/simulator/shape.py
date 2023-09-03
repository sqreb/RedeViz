from __future__ import annotations
import numpy as np
from numpy.random import uniform, randint
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, rotate, translate
from shapely import wkt
from redeviz.simulator.utils import radian2angle

class Shape(object):
    """Boundary shape of cells or nucleuses."""
    def __init__(self, polygon=None) -> None:
        self.polygon = polygon
        self._point_distance_to_boundary_dict = dict()

    def simulate(self, *args, **kwargs):
        raise NotImplementedError()

    def get_simulate_info(self, cfg: dict) -> dict:
        a = uniform(6100, 11000)
        b = uniform(6100, 11000)
        phi = None
        is_steel = False
        if "shape_a" in cfg.keys():
            a = cfg["shape_a"]
        if "shape_b" in cfg.keys():
            b = cfg["shape_b"]
        if "shape_phi" in cfg.keys():
            phi = cfg["shape_phi"]
        if "shape_is_steel" in cfg.keys():
            is_steel = cfg["shape_is_steel"]
        simu_info = {
            "a": a,
            "b": b,
            "phi": phi,
            "cell_radius": max(a, b),
            "is_steel": is_steel
        }
        return simu_info

    def area(self) -> float:
        return self.polygon.area

    def length(self) -> float:
        return self.polygon.length
    
    def centroid(self) -> Point:
        return self.polygon.centroid

    def is_polygon(self) -> bool:
        return self.polygon.type == "Polygon"

    def is_contains(self, other: Shape) -> bool:
        return self.polygon.contains(other.polygon)
    
    def is_intersects(self, other: Shape) -> bool:
        return self.polygon.intersects(other.polygon)

    def distance_to_boundary(self, other: Point|Polygon, resolution=20) -> float:
        if isinstance(other, Point):
            key = (int(other.x/resolution), int(other.y/resolution))
            if key not in self._point_distance_to_boundary_dict.keys():
                self._point_distance_to_boundary_dict[key] = self.polygon.exterior.distance(other)
            dist = self._point_distance_to_boundary_dict[key]
        else:
            dist = self.polygon.exterior.distance(other)
        return dist

    def buffer(self, size: float) -> None:
        self.polygon = self.polygon.buffer(size)
    
    def difference(self, other: Shape) -> None:
        self.polygon = self.polygon.difference(other.polygon)

    def smooth(self, size: float) -> None:
        self.buffer(-1*size)
        self.buffer(size)


class EllipseShape(Shape):
    def simulate(self, mu_X: float, mu_Y: float, a: float, b: float, phi=None, n=512, random_scale=False, **kwargs) -> None:
        if phi is None:
            phi = uniform(0, np.pi)
        if random_scale:
            a = a * uniform(0.9, 1.1)
            b = b * uniform(0.9, 1.1)
        theta_li = np.linspace(0, 2*np.pi, num=n+1)[:n]
        x_li = mu_X + a * np.cos(theta_li)
        y_li = mu_Y + b * np.sin(theta_li)
        ellipse = Polygon([(x, y) for (x, y) in zip(x_li, y_li)])
        ellipse = rotate(ellipse, radian2angle(phi))
        self.polygon = ellipse


norm_fus = Point(-0.5, 0).buffer(1).intersection(Point(0.5, 0).buffer(1))


class FusiformShape(Shape):
    def simulate(self, mu_X: float, mu_Y: float, a: float, b: float, phi: float|None, smooth_factor=1/20, random_scale=False, **kwargs) -> None:
        if phi is None:
            phi = uniform(0, np.pi)
        if random_scale:
            a = a * uniform(0.9, 1.1)
            b = b * uniform(0.9, 1.1)
        fus = translate(rotate(scale(norm_fus, a, 2*b/np.sqrt(3)), radian2angle(phi)), mu_X, mu_Y)
        self.polygon = fus
        self.smooth(min(a, b) * smooth_factor)


class StarShape(Shape):
    def simulate(self, N: int, alpha: float, mu_X: float, mu_Y: float, a: float, b: float, phi: float|None, smooth_factor=1/20, random_scale=False, **kwargs) -> None:
        if phi is None:
            phi = uniform(0, np.pi)
        if random_scale:
            a = a * uniform(0.9, 1.1)
            b = b * uniform(0.9, 1.1)
        theta_li = np.array([index * np.pi / N for index in range(2*N)])
        pho_dict = {0: 1, 1: alpha}
        pho = np.array([pho_dict[index % 2] for index in range(2*N)])
        x_li = mu_X + pho * np.cos(theta_li)
        y_li = mu_Y + pho * np.sin(theta_li)
        norm_star = Polygon([(x, y) for (x, y) in zip(x_li, y_li)])
        star = rotate(scale(norm_star, a, b), radian2angle(phi))
        self.polygon = star
        self.smooth(min(a, b) * smooth_factor)

    def get_simulate_info(self, cfg: dict) -> dict:
        N = randint(4, 7)
        alpha = 0.4
        a = uniform(6100, 11000) * 2
        b = uniform(6100, 11000) * 2
        phi = None
        is_steel = False
        if "shape_N" in cfg.keys():
            N = cfg["shape_N"]
        if "shape_alpha" in cfg.keys():
            alpha = cfg["shape_alpha"]
        if "shape_a" in cfg.keys():
            a = cfg["shape_a"]
        if "shape_b" in cfg.keys():
            b = cfg["shape_b"]
        if "shape_phi" in cfg.keys():
            phi = cfg["shape_phi"]
        if "shape_is_steel" in cfg.keys():
            is_steel = cfg["shape_is_steel"]
        simu_info = {
            "N": N,
            "alpha": alpha,
            "a": a,
            "b": b,
            "phi": phi,
            "cell_radius": max(a, b),
            "is_steel": is_steel
        }
        return simu_info


class RandomShape(Shape):
    def simulate(self, mu_X: float, mu_Y: float, min_R: float, max_R: float, n=512, OrderNum=20, **kwargs) -> None:
        theta_li = np.linspace(0, 2*np.pi, num=n+1)[:n]
        coefficient_li = np.random.randn(OrderNum)
        res_mat = np.zeros([OrderNum, n])
        for index in range(OrderNum):
            res_mat[index, :] = np.cos(index * theta_li + np.random.randn())
        R = np.dot(np.diag(coefficient_li), res_mat).sum(0)
        R_max = R.max()
        R_min = R.min()
        R = ((R - R_min) * (max_R - min_R) / (R_max - R_min)) + min_R
        x_li = mu_X + R * np.cos(theta_li)
        y_li = mu_Y + R * np.sin(theta_li)
        pol = Polygon([(x, y) for (x, y) in zip(x_li, y_li)])
        self.polygon = pol

    def get_simulate_info(self, cfg: dict) -> dict:
        min_R = 7100
        max_R = 12000
        is_steel = False
        if "shape_min_R" in cfg.keys():
            min_R = cfg["shape_min_R"]
        if "shape_max_R" in cfg.keys():
            max_R = cfg["shape_max_R"]
        if "shape_is_steel" in cfg.keys():
            is_steel = cfg["shape_is_steel"]
        simu_info = {
            "min_R": min_R,
            "max_R": max_R,
            "cell_radius": max_R,
            "is_steel": is_steel
        }
        return simu_info


class CustomShape(Shape):
    def simulate(self, polygon: Polygon, mu_X: float, mu_Y: float, a: float, b: float, phi: float, smooth_factor=1/20, random_scale=False, **kwargs) -> None:
        if random_scale:
            a = a * uniform(0.9, 1.1)
            b = b * uniform(0.9, 1.1)
        xmin, ymin, xmax, ymax = polygon.bounds
        self.polygon = translate(rotate(scale(polygon, a/(xmax-xmin), b/(ymax-ymin)), radian2angle(phi)), mu_X, mu_Y)
        self.smooth(min(a, b) * smooth_factor)

    def get_simulate_info(self, cfg: dict) -> dict:
        simu_info = super().get_simulate_info(cfg)
        polygon = wkt.loads(cfg["shape_polygon"])
        assert polygon.type == "Polygon"
        simu_info["polygon"] = polygon
        return simu_info
