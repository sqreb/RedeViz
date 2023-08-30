from __future__ import annotations
from numpy.random import uniform, normal, randint
import numpy as np
import pandas as pd
import random
import os
import logging
from collections import defaultdict
from typing import List
from shapely.ops import voronoi_diagram, clip_by_rect
from shapely.geometry import MultiPoint, box, MultiPolygon, Point, Polygon
from shapely.prepared import prep
from shapely import wkt
from plotnine import *
from redeviz.simulator.shape import EllipseShape, FusiformShape, StarShape, RandomShape, CustomShape
from redeviz.simulator.layout import Layout, RandomLayout, RectangleLayout, TriangleLayout, ParallelogramLayout, HexagonalLayout, CustomLayout
from redeviz.simulator.utils import uniform2D, compute_pairwise_dist, iter_uniform2D, iter_uniform
from redeviz.simulator.gene_expr import GeneExprInfo

FORMAT = '%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)

class DivInfo(object):
    # ALL_DIV_TYPE = ["Normal", "Ellipse", "SingleCell", "Custom"]
    ALL_DIV_TYPE = ["Normal", "SingleCell", "Custom"]
    def __init__(self, cell_type: str, cfg: dict) -> None:
        self.cell_type = cell_type
        if "div_type" in cfg.keys():
            self.div_type = cfg["div_type"]
        else:
            self.div_type = "Normal"
        if self.div_type not in self.ALL_DIV_TYPE:
            raise ValueError(f'{cell_type}: div_type should be in {self.ALL_DIV_TYPE}')
        if "div_num" in cfg.keys():
            self.div_num = cfg["div_num"]
        else:
            if self.div_type == "Normal":
                self.div_num = 2
            elif self.div_type == "Ellipse":
                self.div_num = 4
            elif self.div_type == "SingleCell":
                self.div_num = 20

        self.div_radius = None
        if self.div_type == "Ellipse":
            if "div_radius" in cfg.keys():
                self.div_radius = cfg["div_radius"]
            else:
                self.div_radius = 40000
        if self.div_type == "Custom":
            self.polygons = wkt(cfg["div_region"])
        else:
            self.polygons = None


class LayoutInfo(object):
    LAYOUT_DICT = {
        "Random": RandomLayout, 
        "Rectangle": RectangleLayout,
        "Triangle": TriangleLayout,
        "Parallelogram": ParallelogramLayout,
        "Hexagonal": HexagonalLayout,
        "Custom": CustomLayout
        }
    def __init__(self, cell_type: str, cfg: dict, x_range: float, y_range: float, nucleis_radius: float) -> None:
        self.cell_type = cell_type
        self.x_range = x_range
        self.y_range = y_range
        self.nucleis_radius = nucleis_radius
        self.layout_type = cfg["layout_type"]
        if self.layout_type not in self.LAYOUT_DICT.keys():
            raise ValueError(f'{cell_type}: layout_type should be in {self.LAYOUT_DICT.keys()}')
        self.layout_info = self.LAYOUT_DICT[self.layout_type].get_simulate_info(cfg)
        self.layout = None

    def simulate_layout_in_region(self, polygon: Polygon):
        if self.layout_type == "Custom":
            self.layout = self.LAYOUT_DICT[self.layout_type](self.x_range, self.y_range, self.nucleis_radius)
            self.layout.simulate(**self.layout_info)
        else:
            if polygon.geom_type == "MultiPolygon":
                cell_map_li = list()
                for tmp_polygon in polygon.geoms:
                    xmin, ymin, xmax, ymax = tmp_polygon.bounds
                    layout = self.LAYOUT_DICT[self.layout_type](xmax-xmin, ymax-ymin, self.nucleis_radius)
                    layout.simulate(**self.layout_info)
                    layout.translate(xmin, ymin)
                    cell_map_li += list(layout.cell_map.geoms)
                xmin, ymin, xmax, ymax = polygon.bounds
                self.layout = self.LAYOUT_DICT[self.layout_type](xmax-xmin, ymax-ymin, self.nucleis_radius)
                self.layout.cell_map = MultiPoint(cell_map_li)
            else:
                xmin, ymin, xmax, ymax = polygon.bounds
                self.layout = self.LAYOUT_DICT[self.layout_type](xmax-xmin, ymax-ymin, self.nucleis_radius)
                self.layout.simulate(**self.layout_info)
                self.layout.translate(xmin, ymin)
        self.layout.clip(polygon)


class ShapeInfo(object):
    SHAPE_DICT = {
        "Ellipse": EllipseShape,
        "Fusiform": FusiformShape,
        "Star": StarShape,
        "Random": RandomShape,
        "Custom": CustomShape
    }
    def __init__(self, cell_type: str, cfg: dict) -> None:
        self.cell_type = cell_type
        self.shape_type = cfg["shape_type"]
        if self.shape_type not in self.SHAPE_DICT.keys():
            raise ValueError(f'{cell_type}: shape_type should be in {self.SHAPE_DICT.keys()}')
        self.shape = self.SHAPE_DICT[self.shape_type]()
        self.shape_info = self.shape.get_simulate_info(cfg)
        is_steel = False
        if "shape_is_steel" in cfg.keys():
            is_steel = cfg["shape_is_steel"]
        self.is_steel = is_steel


class PointNumInfo(object):
    def __init__(self, cell_type: str, cfg: dict) -> None:
        self.cell_type = cell_type
        if "RNA_per_cell_mu" in cfg.keys():
            self.RNA_per_cell_mu = cfg["RNA_per_cell_mu"]
        else:
            self.RNA_per_cell_mu = 3000
        if "RNA_per_cell_log2sd" in cfg.keys():
            self.RNA_per_cell_log2sd = cfg["RNA_per_cell_log2sd"]
        else:
            self.RNA_per_cell_log2sd = 0.2
    
    def simulate_mu_point_num(self) -> float:
        return 2 ** (normal(np.log2(self.RNA_per_cell_mu), self.RNA_per_cell_log2sd))

class CellMapInfo(object):
    def __init__(self, cfg: dict) -> None:
        if "x_range" in cfg.keys():
            self.x_range = cfg["x_range"]
        else:
            self.x_range = 1e6

        if "y_range" in cfg.keys():
            self.y_range = cfg["y_range"]
        else:
            self.y_range = 1e6

        if "nucleis_radius" in cfg.keys():
            self.nucleis_radius = cfg["nucleis_radius"]
            assert len(self.nucleis_radius) == 2
        else:
            self.nucleis_radius = [2500, 5000]

        if "nucleus_shift_factor" in cfg.keys():
            self.nucleus_shift_factor = cfg["nucleus_shift_factor"]
        else:
            self.nucleus_shift_factor = 0.2
        
        if "nucleus_RNA_capture_efficiency" in cfg.keys():
            self.nucleus_RNA_capture_efficiency = cfg["nucleus_RNA_capture_efficiency"]
        else:
            self.nucleus_RNA_capture_efficiency = 1
        
        if "uniform_noise_expr_ratio" in cfg.keys():
            self.uniform_noise_expr_ratio = cfg["uniform_noise_expr_ratio"]
        else:
            self.uniform_noise_expr_ratio = 0.0001

        if "diffusion_noise_expr_ratio" in cfg.keys():
            self.diffusion_noise_expr_ratio = cfg["diffusion_noise_expr_ratio"]
        else:
            self.diffusion_noise_expr_ratio = 0.01

        # self.gap_size = cfg["gap_size"]
        # self.spot_size = cfg["spot_size"]
        # self.UMI_per_spot = cfg["UMI_per_spot"]

        cell_info_cfg = cfg["cell_info"]
        self.cell_type = list(cell_info_cfg.keys())
        self.expr_info = GeneExprInfo(cfg["ref_expr"], self.cell_type)
        self.div_info = {cell_type: DivInfo(cell_type, info) for (cell_type, info) in cell_info_cfg.items()}
        self.layout_info = {cell_type: LayoutInfo(cell_type, info, self.x_range, self.y_range, self.nucleis_radius[1]) for (cell_type, info) in cell_info_cfg.items()}
        self.shape_info = {cell_type: ShapeInfo(cell_type, info) for (cell_type, info) in cell_info_cfg.items()}
        self.point_num_info = {cell_type: PointNumInfo(cell_type, info) for (cell_type, info) in cell_info_cfg.items()}

class Cell(object):
    def __init__(self, name: str, x: float, y: float, cell_type: str, shape_info: ShapeInfo, point_num_info: PointNumInfo) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.cell_type = cell_type
        self.shape_info = shape_info
        self.cell_shape = None
        self.nucleus_shape = None
        self.points = None
        self.nucleus_radius = None
        self.adj_cell_li = list()
        self.tmp_cell_x_coords = None
        self.tmp_cell_y_coords = None
        self.tmp_prep_cell_shape = None
        self.norm_fct = None
        self.stop_update_li = None
        self.finish_update = False
        self.RNA_point_li = list()
        self.mu_RNA_point = point_num_info.simulate_mu_point_num()
        self._is_in_cell_dict = dict()
        self._is_in_nucleus_dict = dict()
        self._cell_shape_prep = None
        self._nucleus_shape_prep = None

    def is_overlap(self, other: Cell) -> bool:
        return self.cell_shape.is_intersects(other.cell_shape)

    def is_polygon_shape(self) -> bool:
        return self.cell_shape.is_polygon()

    @classmethod
    def init_is_in_polygon(cls, polygon: Polygon, resolution=20):
        mid_point_coord = np.array(polygon.centroid.xy)
        bound_coord = np.array(polygon.exterior.xy)
        diff_coord = bound_coord - mid_point_coord
        xmin, ymin, xmax, ymax = polygon.bounds
        bound_range = max(xmax-xmin, ymax-ymin)
        ratio_step = resolution / bound_range
        res_dict = dict()
        for ratio in np.arange(0, 1+ratio_step, ratio_step):
            coord = mid_point_coord + ratio * diff_coord
            for index in range(coord.shape[1]):
                x, y = coord[:, index]
                res_dict[int(x/resolution), int(y/resolution)] = True
        for ratio in np.arange(1+ratio_step, 2, ratio_step):
            coord = mid_point_coord + ratio * diff_coord
            for index in range(coord.shape[1]):
                x, y = coord[:, index]
                res_dict[int(x/resolution), int(y/resolution)] = False
        return res_dict

    def is_in_cell(self, point: Point, resolution=20) -> bool:
        if self._cell_shape_prep is None:
            self._cell_shape_prep = prep(self.cell_shape.polygon)
        key = (int(point.x/resolution), int(point.y/resolution))
        if key not in self._is_in_cell_dict.keys():
            self._is_in_cell_dict[key] = self._cell_shape_prep.contains(point)
        return self._is_in_cell_dict[key]
    
    def is_in_nucleus(self, point: Point, resolution=20) -> bool:
        if self._nucleus_shape_prep is None:
            self._nucleus_shape_prep = prep(self.nucleus_shape.polygon)
        xmin, ymin, xmax, ymax = self.nucleus_shape.polygon.bounds
        if (point.x < xmin) | (point.x > xmax):
            return False
        if (point.y < ymin) | (point.y > ymax):
            return False
        key = (int(point.x/resolution), int(point.y/resolution))
        if key not in self._is_in_nucleus_dict.keys():
            self._is_in_nucleus_dict[key] = self._nucleus_shape_prep.contains(point)
        return self._is_in_nucleus_dict[key]

    def simulate_ref_shape(self, nucleus_radius=(2500, 5000), nucleus_shift_factor=0.2) -> None:
        while True:
            tmp_nucleus_radius = uniform(nucleus_radius[0], nucleus_radius[1], 2)
            nucleus_x_coord = self.x + tmp_nucleus_radius[0] * uniform(-1*nucleus_shift_factor, nucleus_shift_factor)
            nucleus_y_coord = self.y + tmp_nucleus_radius[1] * uniform(-1*nucleus_shift_factor, nucleus_shift_factor)
            nucleus_coord = np.array([nucleus_x_coord, nucleus_y_coord])
            nucleus_phi = uniform(0, 2*np.pi, 1)
            nucleus_shape = EllipseShape()
            nucleus_shape.simulate(nucleus_coord[0], nucleus_coord[1], tmp_nucleus_radius[0], tmp_nucleus_radius[1], nucleus_phi)

            cell_shape = self.shape_info.SHAPE_DICT[self.shape_info.shape_type]()
            cell_shape.simulate(mu_X=self.x, mu_Y=self.y, random_scale=True, **self.shape_info.shape_info)
            if not cell_shape.is_polygon():
                continue

            if cell_shape.is_contains(nucleus_shape):
                self.cell_shape = cell_shape
                self.nucleus_shape = nucleus_shape
                self.nucleus_radius = nucleus_radius
                return None

    def init_shape(self, adj_cell_li: List[Cell]) -> None:
        self.adj_cell_li = adj_cell_li
        ori_cell_x, ori_cell_y = self.cell_shape.polygon.exterior.xy
        ori_cell_x = np.array(ori_cell_x)
        ori_cell_y = np.array(ori_cell_y)
        if self.shape_info.shape_info["is_steel"]:
            self.tmp_cell_x_coords = ori_cell_x
            self.tmp_cell_y_coords = ori_cell_y
            self.ori_cell_delta_x = np.zeros_like(ori_cell_x)
            self.ori_cell_delta_y = np.zeros_like(ori_cell_y)
            self.tmp_prep_cell_shape = prep(self.cell_shape.polygon)
            self.stop_update_li = np.array([True] * len(self.tmp_cell_x_coords))
            self.norm_fct = 1
        else:
            nucleus_center_x, nucleus_center_y = self.nucleus_shape.polygon.centroid.xy
            self.ori_cell_delta_x = ori_cell_x - nucleus_center_x
            self.ori_cell_delta_y = ori_cell_y - nucleus_center_y
            self.norm_fct = 0.3
            self.tmp_cell_x_coords = nucleus_center_x +  self.ori_cell_delta_x * self.norm_fct
            self.tmp_cell_y_coords = nucleus_center_y +  self.ori_cell_delta_y * self.norm_fct
            self.tmp_prep_cell_shape = prep(Polygon(np.array([self.tmp_cell_x_coords, self.tmp_cell_y_coords]).T))
            self.stop_update_li = np.array([False] * len(self.tmp_cell_x_coords))

    @classmethod
    def compute_force_arr(cls, x_coords: np.ndarray, y_coords: np.ndarray, ori_dist_mat: np.ndarray, x2_coords=None, y2_coords=None):
        if x2_coords is None:
            x_diff_mat = x_coords[:, np.newaxis] - x_coords
            y_diff_mat = y_coords[:, np.newaxis] - y_coords
        else:
            x_diff_mat = x_coords[:, np.newaxis] - x2_coords
            y_diff_mat = y_coords[:, np.newaxis] - y2_coords
        dist_mat = np.sqrt(x_diff_mat ** 2 + y_diff_mat ** 2)
        delta_dist_mat = dist_mat - ori_dist_mat
        e_x_mat = x_diff_mat / (dist_mat+1e-3)
        e_y_mat = y_diff_mat / (dist_mat+1e-3)
        force_x_arr = (e_x_mat * delta_dist_mat).sum(1)
        force_y_arr = (e_y_mat * delta_dist_mat).sum(1)
        return force_x_arr, force_y_arr

    @classmethod
    def smooth(cls, x_li: np.ndarray):
        x_li = list(x_li)
        x_len = len(x_li)
        new_li = 2*np.array(x_li) + np.array([x_li[-1]] + x_li[:(x_len-1)]) + np.array(x_li[1:]+[x_li[0]])
        return new_li / 4

    def update_shape(self, step=0.1) -> None:
        ## TODO: Need to speed up
        ## Init. points & contain method
        if self.finish_update:
            return None
        if self.adj_cell_li:
            for index in range(len(self.stop_update_li)):
                if not self.stop_update_li[index]:
                    point = Point(self.tmp_cell_x_coords[index], self.tmp_cell_y_coords[index])
                    for adj_cell in self.adj_cell_li:
                        if adj_cell.tmp_prep_cell_shape.contains(point):
                            self.stop_update_li[index] = True
                            break
        if self.stop_update_li.all():
            self.finish_update = True
            return None

        delta_x_arr = step * self.ori_cell_delta_x
        delta_y_arr = step * self.ori_cell_delta_y
        delta_x_arr[self.stop_update_li] = 0
        delta_y_arr[self.stop_update_li] = 0
        new_cell_x_coords = self.tmp_cell_x_coords + delta_x_arr
        new_cell_y_coords = self.tmp_cell_y_coords + delta_y_arr
        new_cell_x_coords = self.smooth(new_cell_x_coords)
        new_cell_y_coords = self.smooth(new_cell_y_coords)
        self.norm_fct += step
        is_finish = self.norm_fct >= 1
        if is_finish:
            self.finish_update = True
            self.stop_update_li = np.array([True]*new_cell_x_coords)
        self.tmp_cell_x_coords = new_cell_x_coords
        self.tmp_cell_y_coords = new_cell_y_coords
        self.tmp_prep_cell_shape = prep(Polygon(np.array([self.tmp_cell_x_coords, self.tmp_cell_y_coords]).T))

    def buffer(self, size: float) -> None:
        self.cell_shape.buffer(size)
    
    def adjust_cell_bondary(self, other: Cell) -> None:
        if self.cell_shape.is_intersects(other.nucleus_shape):
            self.cell_shape.polygon = self.cell_shape.polygon.difference(other.nucleus_shape.polygon.buffer(other.nucleus_radius[0]*0.2))
        if other.cell_shape.is_intersects(self.nucleus_shape):
            other.cell_shape.polygon = other.cell_shape.polygon.difference(self.nucleus_shape.polygon.buffer(other.nucleus_radius[0]*0.2))
        
        new_pol = self.cell_shape.polygon.difference(other.cell_shape.polygon)
        if new_pol.type == 'Polygon':
            self.cell_shape.polygon = new_pol
        else:
            for pol in new_pol.geoms:
                if pol.contains(Point(self.x, self.y)):
                    self.cell_shape.polygon = pol
                    return None

    @classmethod
    def cytosol_point_sampling_prob(cls, is_in_nucleus, dist, NCR, rho):
        if dist >= rho:
            p = 1
        else:
            p = np.cos((dist/rho-1) * np.pi / 2)

        if is_in_nucleus:
            return 1 - (1-NCR) * p
        else:
            return p

    @classmethod
    def nucleus_point_sampling_prob(cls, is_in_nucleus, dist, NCR, rho):
        if dist >= rho:
            p = 1
        else:
            p = np.cos((dist/rho-1) * np.pi / 2)

        if is_in_nucleus:
            return p + (1-p)/NCR
        else:
            return p / NCR

    @classmethod
    def cell_point_sampling_prob(cls, is_in_nucleus, dist, rho):
        if is_in_nucleus:
            return 1
        if dist >= rho:
            return 1
        else:
            return np.cos((dist/rho-1) * np.pi / 2)

    def simulate_gene_point_number(self, gene_expr_info: GeneExprInfo, N: int):
        return gene_expr_info.simulate_gene_point_number(self.cell_type, N)

    def simulate_RNA_in_cell(self, gene_expr_info: GeneExprInfo, nucleus_RNA_capture_efficiency: float):
        # TODO: speedup
        # is_in_cell, distance_to_boundary
        gene_umi_li = self.simulate_gene_point_number(gene_expr_info, self.mu_RNA_point)
        xmin, ymin, xmax, ymax = self.cell_shape.polygon.bounds
        point_iter = iter_uniform2D(xmin, xmax, ymin, ymax)
        p_iter = iter_uniform(0, 1)
        rho = np.sqrt(self.nucleus_shape.polygon.area) / 20
        RNA_point_li = list()
        # self._is_in_cell_dict = self.init_is_in_polygon(self.cell_shape.polygon)
        # self._is_in_nuclues_dict = self.init_is_in_polygon(self.nucleus_shape.polygon)
        for (gid, num, NCR) in gene_umi_li:
            index = 0
            while index < num:
                x, y = next(point_iter)
                point = Point(x, y)
                if not self.is_in_cell(point):
                    continue
                is_in_nucleus = self.is_in_nucleus(point)

                # if is_in_nucleus:
                #     dist = self.nucleus_shape.distance_to_boundary(point)
                # else:
                #     dist = self.cell_shape.distance_to_boundary(point)
                dist = rho
                if NCR==1:
                    p_cutoff = self.cell_point_sampling_prob(is_in_nucleus, dist, rho)
                elif NCR > 1:
                    p_cutoff = self.nucleus_point_sampling_prob(is_in_nucleus, dist, NCR, rho)
                else:
                    p_cutoff = self.cytosol_point_sampling_prob(is_in_nucleus, dist, NCR, rho)
                if next(p_iter) > p_cutoff:
                    continue
                index += 1
                if is_in_nucleus and (next(p_iter) > nucleus_RNA_capture_efficiency):
                    continue
                if is_in_nucleus:
                    loc = "Nucleus"
                else:
                    loc = "Cytosol"
                RNA_point_li.append((self.name, gid, x, y, loc))
        self.RNA_point_li = RNA_point_li

    def simulate_RNA_around_cell(self, gene_expr_info: GeneExprInfo, bg_expr_fct: float):
        gene_umi_li = self.simulate_gene_point_number(gene_expr_info, self.mu_RNA_point*bg_expr_fct)
        sd = np.sqrt(self.cell_shape.polygon.area)

        def iter_point(mu_x, mu_y, polygon, sd, buffer_size=10000):
            bound_x, bound_y = polygon.boundary.xy
            delta_x = np.array(bound_x) - mu_x
            delta_y = np.array(bound_y) - mu_y
            delta_y[delta_y==0] = 1e-3
            base_dist = np.sqrt(delta_x**2+delta_y**2)
            phi = np.arctan(delta_y/delta_x)
            phi[delta_x<0] += np.pi
            while True:
                delta_dist = np.abs(normal(0, sd, buffer_size))
                tmp_index_li = np.random.choice(list(range(len(base_dist))), buffer_size, replace=True)
                x_rand_li = uniform(-10, 10, buffer_size)
                y_rand_li = uniform(-10, 10, buffer_size)
                for index, (tmp_index, tmp_delta_dist) in enumerate(zip(tmp_index_li, delta_dist)):
                    tmp_phi = phi[tmp_index]
                    tmp_dist = base_dist[tmp_index] + tmp_delta_dist
                    x = mu_x + tmp_dist * np.cos(tmp_phi) + x_rand_li[index]
                    y = mu_y + tmp_dist * np.sin(tmp_phi) + y_rand_li[index]
                    yield (x, y)

        point_iter = iter_point(self.x, self.y, self.cell_shape.polygon, sd)
        RNA_point_li = list()
        for (gid, num, _) in gene_umi_li:
            index = 0
            while index < num:
                x, y = next(point_iter)
                point = Point(x, y)
                if self.is_in_cell(point):
                    continue
                index += 1
                RNA_point_li.append((self.name, gid, x, y, "Diffusion"))
        return RNA_point_li

class CellMap(object):
    def __init__(self, map_info: CellMapInfo) -> None:
        self.map_info = map_info
        self.cell_li = list()
        self.cell_cell_interaction_li = list()
        self.RNA_point_df = None
    
    def simulate_plane_segmentation(self) -> None:
        normal_div_num_dict = {cell_type: div_info.div_num for (cell_type, div_info) in self.map_info.div_info.items() if div_info.div_type=="Normal"}
        total_normal_div_num = sum(normal_div_num_dict.values())
        if not total_normal_div_num:
            return None
        p_iter = iter_uniform(0, 1)
        while True:
            normal_div_mid_points = MultiPoint(uniform2D(0, self.map_info.x_range, 0, self.map_info.y_range, total_normal_div_num))
            region_boundary = box(0, 0, self.map_info.x_range, self.map_info.y_range)
            normal_regions = list(voronoi_diagram(normal_div_mid_points, envelope=region_boundary))
            for index in range(total_normal_div_num):
                normal_regions[index] = clip_by_rect(normal_regions[index], 0, 0, self.map_info.x_range, self.map_info.y_range)
                xmin, ymin, xmax, ymax = normal_regions[index].bounds
                size = min(xmax-xmin, ymax-ymin)
                normal_regions[index] = normal_regions[index].buffer(-1*size/20).buffer(size/10)
            for index1 in range(total_normal_div_num):
                for index2 in range(index1):
                    region1 = normal_regions[index1]
                    region2 = normal_regions[index2]
                    if region1.intersects(region2):
                        if next(p_iter) > 0.5:
                            normal_regions[index1] = region1.difference(region2)
                        else:
                            normal_regions[index2] = region2.difference(region1)
            for index in range(total_normal_div_num):
                normal_regions[index] = clip_by_rect(normal_regions[index], 0, 0, self.map_info.x_range, self.map_info.y_range)
                xmin, ymin, xmax, ymax = normal_regions[index].bounds
                size = min(xmax-xmin, ymax-ymin)
                normal_regions[index] = normal_regions[index].buffer(-1*size/10).buffer(size/10)
            is_failed = False
            for region in normal_regions:
                if region.area == 0:
                    is_failed = True
                    break
                if region.type != "Polygon":
                    is_failed = True
                    break
                if not region.is_valid:
                    is_failed = True
                    break
            if is_failed:
                continue
            random.shuffle(normal_regions)
            index = 0
            for cell_type, div_num in normal_div_num_dict.items():
                self.map_info.div_info[cell_type].polygons = MultiPolygon(normal_regions[index: index+div_num])
                index += div_num
            return None

    def simulate_ellipse_regions(self) -> None:
        ellipse_cell_types = [cell_type for (cell_type, div_info) in self.map_info.div_info.items() if div_info.div_type=="Ellipse"]
        circ_num = len(ellipse_cell_types)
        if not circ_num:
            return None

        while True:
            for cell_type in ellipse_cell_types:
                div_info = self.map_info.div_info[cell_type]
                mid_points = RandomLayout(self.map_info.x_range, self.map_info.y_range)
                mid_points.simulate(div_info.div_num, div_info.div_radius)
                polygon_li = list()
                for point in mid_points.cell_map.geoms:
                    ellipse = EllipseShape()
                    ellipse.simulate(point.x, point.y, div_info.div_radius, uniform(0.5, 0.8)*div_info.div_radius, random_scale=True)
                    polygon_li.append(ellipse.polygon)
                div_info.polygons = MultiPolygon(polygon_li)
            is_failed = False
            for index1 in range(circ_num):
                if is_failed:
                    break
                polygon1 = self.map_info.div_info[ellipse_cell_types[index1]].polygons
                for index2 in range(index1):
                    polygon2 = self.map_info.div_info[ellipse_cell_types[index2]].polygons
                    if polygon1.intersects(polygon2):
                        is_failed = True
                        break
            if is_failed:
                continue
            return None

    def adjust_cell_coords(self, cell_li: List[Cell], max_iter=10000) -> List[Cell]:
        cell_coords = np.array([(cell.x, cell.y) for cell in cell_li])
        radius = self.map_info.nucleis_radius[1]*1.2
        for index in range(max_iter):
            cell_coords, max_rest_dist, _ = Layout.update_coords(cell_coords, radius, 0, self.map_info.x_range, 0, self.map_info.y_range)
            if max_rest_dist == 0:
                break
        for index in range(len(cell_li)):
            cell_li[index].x = cell_coords[index, 0]
            cell_li[index].y = cell_coords[index, 1]
        return cell_li

    def layout_simulation(self):
        div_type_dict = defaultdict(list)
        for (cell_type, div_info) in self.map_info.div_info.items():
            div_type_dict[div_info.div_type].append(cell_type)

        for _ in range(1000):
            try:
                self.simulate_plane_segmentation()
            except ValueError:
                continue

            # self.simulate_ellipse_regions()
            is_failed = False
            for normal_name in div_type_dict["Normal"]:
                normal_info = self.map_info.div_info[normal_name]
                for ellipse_name in div_type_dict["Ellipse"]:
                    ellipse_info = self.map_info.div_info[ellipse_name]
                    if normal_info.polygons.intersects(ellipse_info.polygons):
                        normal_info.polygons = normal_info.polygons.difference(ellipse_info.polygons)
                if normal_info.polygons.area == 0:
                    is_failed = True
                    break
            if is_failed:
                continue

            for custom_name in div_type_dict["Custom"]:
                custom_info = self.map_info.div_info[custom_name]
                for other_name in div_type_dict["Normal"] + div_type_dict["Ellipse"]:
                    other_info = self.map_info.div_info[other_name]
                    if custom_info.polygons.intersects(other_info.polygons):
                        other_info.polygons = other_info.polygons.difference(custom_info.polygons)
            is_failed = False
            for other_name in div_type_dict["Normal"] + div_type_dict["Ellipse"]:
                other_info = self.map_info.div_info[other_name]
                if other_info.polygons.area == 0:
                    is_failed = True
                    break
            if is_failed:
                continue

            cell_index = 0
            cell_li = list()
            for cell_type in div_type_dict["Normal"] + div_type_dict["Ellipse"] + div_type_dict["Custom"]:
                div_info = self.map_info.div_info[cell_type]
                layout_info = self.map_info.layout_info[cell_type]
                layout_info.simulate_layout_in_region(div_info.polygons)
                shape_info = self.map_info.shape_info[cell_type]
                point_num_info = self.map_info.point_num_info[cell_type]
                cell_li += [Cell(cell_index+index, point.x, point.y, cell_type, shape_info, point_num_info) for (index, point) in enumerate(layout_info.layout.cell_map.geoms)]
                cell_index += len(layout_info.layout.cell_map.geoms)
            random.shuffle(cell_li)

            cell_index = 0
            cell_num = len(cell_li)
            for cell_type in div_type_dict["SingleCell"]:
                div_info = self.map_info.div_info[cell_type]
                shape_info = self.map_info.shape_info[cell_type]
                for index in range(cell_index, cell_index+div_info.div_num):
                    if index >= cell_num:
                        raise ValueError(f"{cell_type}: Cell number is too large.")
                    cell_li[index].cell_type = cell_type
                    cell_li[index].shape_info = shape_info
                cell_index += div_info.div_num
            cell_li = self.adjust_cell_coords(cell_li)
            self.cell_li = cell_li
            is_fail = False
            return is_fail
        is_fail = True
        return is_fail

    def shape_simulation(self, smooth_fct=0.3, membrane_size=10, max_step=1000):
        cell_num = len(self.cell_li)
        p_iter = iter_uniform(0, 1)
        for _ in range(2):
            is_fail = False
            for cell in self.cell_li:
                cell.simulate_ref_shape(self.map_info.nucleis_radius, self.map_info.nucleus_shift_factor)

            x_coords = np.array([cell.x for cell in self.cell_li])
            y_coords = np.array([cell.y for cell in self.cell_li])
            cell_radius_li = np.array([cell.shape_info.shape_info["cell_radius"] for cell in self.cell_li])
            dist_square_mat = compute_pairwise_dist(x_coords, y_coords, return_square=True) + 4 * cell_radius_li.max()
            dist_square_cutoff_mat = (cell_radius_li[:, np.newaxis] + cell_radius_li) ** 2
            cell_cell_adj_mat = dist_square_mat <= dist_square_cutoff_mat

            for index in range(cell_num):
                adj_cell_li = [self.cell_li[index2] for index2 in range(cell_num) if cell_cell_adj_mat[index2, index] and (index2 != index)]
                self.cell_li[index].init_shape(adj_cell_li)
            
            steel_cell_li = [cell for cell in self.cell_li if cell.shape_info.shape_info["is_steel"]]
            is_overlap = False
            for index1 in range(len(steel_cell_li)):
                for index2 in range(index1):
                    if steel_cell_li[index1].is_overlap(steel_cell_li[index2]):
                        is_overlap = True
                        is_fail = True
                        break
                if is_overlap:
                    break
            if is_overlap:
                continue

            retain_cell_index_li = list()
            for index in range(cell_num):
                is_overlap = False
                cell = self.cell_li[index]
                if cell.shape_info.shape_info["is_steel"]:
                    retain_cell_index_li.append(index)
                    continue
                for steel_cell in steel_cell_li:
                    if steel_cell.cell_shape.is_intersects(cell.nucleus_shape):
                        is_overlap = True
                        break
                if not is_overlap:
                    retain_cell_index_li.append(index)
            self.cell_li = [self.cell_li[index] for index in retain_cell_index_li]
            cell_num = len(self.cell_li)

            for _ in range(max_step):
                for index in range(cell_num):
                    self.cell_li[index].update_shape()
                if all([cell.finish_update for cell in self.cell_li]):
                    break

            for index in range(cell_num):
                self.cell_li[index].cell_shape.polygon = Polygon([(x, y) for (x, y) in zip(self.cell_li[index].tmp_cell_x_coords, self.cell_li[index].tmp_cell_y_coords)])
                self.cell_li[index].buffer(-1*smooth_fct*self.cell_li[index].nucleus_radius[1])
                self.cell_li[index].buffer(smooth_fct*self.cell_li[index].nucleus_radius[1])

            cell_cell_interaction_li = list()
            for index1 in range(cell_num):
                cell1 = self.cell_li[index1]
                for index2 in range(index1):
                    cell2 = self.cell_li[index2]
                    if cell1.is_overlap(cell2):
                        cell_cell_interaction_li.append((cell1.name, cell2.name))
                        if (not cell1.shape_info.shape_info["is_steel"]) and (not cell2.shape_info.shape_info["is_steel"]):
                            if next(p_iter) < 0.5:
                                cell1.adjust_cell_bondary(cell2)
                            else:
                                cell2.adjust_cell_bondary(cell1)
                        elif cell1.shape_info.shape_info["is_steel"]:
                            cell2.adjust_cell_bondary(cell1)
                        else:
                            cell1.adjust_cell_bondary(cell2)
            self.cell_cell_interaction_li = cell_cell_interaction_li

            for index in range(cell_num):
                self.cell_li[index].buffer(-1*membrane_size)
                if not self.cell_li[index].is_polygon_shape:
                    is_fail = True
                    continue
            return is_fail
        return is_fail

    def RNA_point_simulation(self):
        all_RNA_point_li = list()
        cell_num = len(self.cell_li)
        for index, cell in enumerate(self.cell_li):
            if index % 10 == 0:
                logging.info(f"Simulating RNA points ({index} / {cell_num})")
            cell.simulate_RNA_in_cell(self.map_info.expr_info, self.map_info.nucleus_RNA_capture_efficiency)
            all_RNA_point_li += cell.RNA_point_li
            diffusion_li = cell.simulate_RNA_around_cell(self.map_info.expr_info, self.map_info.diffusion_noise_expr_ratio)
            all_RNA_point_li += diffusion_li
            
        all_area = self.map_info.x_range * self.map_info.y_range
        total_cell_area = sum([cell.cell_shape.polygon.area for cell in self.cell_li])
        total_cell_point = sum([cell.mu_RNA_point for cell in self.cell_li])
        N_bg = total_cell_point / total_cell_area * all_area * self.map_info.uniform_noise_expr_ratio
        point_iter = iter_uniform2D(0, self.map_info.x_range, 0, self.map_info.y_range)
        bg_li = list()
        for (gid, num, _) in self.map_info.expr_info.simulate_ave_point_number(N_bg):
            index = 0
            while index < num:
                x, y = next(point_iter)
                bg_li.append(("UniformNoise", gid, x, y, "UniformNoise"))
                index += 1
        self.RNA_point_df = pd.DataFrame(all_RNA_point_li+bg_li, columns=["CellName", "Gid", "x", "y", "Location"])

    def simulate_all(self):
        logging.info("Simulating layout and cells")
        for index in range(10):
            logging.info(f"Round: {index}")
            is_fail = self.layout_simulation()
            if is_fail:
                raise ValueError("Please check layout configure")
            is_fail = self.shape_simulation()
            if is_fail:
                continue
            break
        if is_fail:
            raise ValueError("Please check shape configure")
        logging.info("Simulating RNA points")
        self.RNA_point_simulation()

    def write_dataset(self, f_dir):
        logging.info("Writting spatial dataset ...")
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        cell_num = len(self.cell_li)
        x_li = list()
        y_li = list()
        cell_type_li = list()
        cell_name_li = list()
        cell_shape_li = list()
        cell_area_li = list()
        nucleus_shape_li = list()
        nucleus_area_li = list()
        mu_point_num_li = list()
        for cell in self.cell_li:
            cell_name_li.append(cell.name)
            cell_shape_li.append(cell.cell_shape.polygon.wkt)
            cell_area_li.append(cell.cell_shape.polygon.area)
            nucleus_shape_li.append(cell.nucleus_shape.polygon.wkt)
            nucleus_area_li.append(cell.nucleus_shape.polygon.area)
            x_li.append(cell.x)
            y_li.append(cell.y)
            cell_type_li.append(cell.cell_type)
            mu_point_num_li.append(cell.mu_RNA_point)
        cell_df = pd.DataFrame({
            "CellName": cell_name_li,
            "CellType": cell_type_li,
            "x": x_li,
            "y": y_li,
            "CellShape": cell_shape_li,
            "CellArea": cell_area_li,
            "NucleusShape": nucleus_shape_li,
            "NucleusArea": nucleus_area_li,
            "ExpPointNum": mu_point_num_li
        })
        cell_df[["CellName"]] = cell_df[["CellName"]].astype("int")
        cell_df = cell_df.sort_values(by="CellName")
        cell_df[["CellName"]] = cell_df[["CellName"]].astype("str")
        cell_df.to_csv(os.path.join(f_dir, "CellShapeInfo.tsv"), sep="\t", index_label=False, index=False)

        div_df_li = list()
        with open(os.path.join(f_dir, "DivShape.tsv"), "w") as f:
            f.write("DivName\tDivShape\n")
            for cell_type, div_info in self.map_info.div_info.items():
                if div_info.div_type not in ["Normal", "Custom"]:
                    continue
                for index, polygon in enumerate(div_info.polygons.geoms):
                    f.write(f"{cell_type}:{index}\t{polygon.wkt}\n")
                    polygon_x, polygon_y = polygon.boundary.xy
                    div_df_li.append(
                        pd.DataFrame({
                            "CellType": cell_type,
                            "RegionIndex": f"{cell_type}:{index}",
                            "x": polygon_x,
                            "y": polygon_y
                        })
                    )
        div_df = pd.concat(div_df_li)

        p = ggplot() + \
            geom_polygon(data=div_df, mapping=aes(x="x", y="y", color="CellType", fill="CellType", group="RegionIndex"), alpha=0.2, size=0.2) + \
            geom_point(data=cell_df, mapping=aes(x="x", y="y", color="CellType"), size=0.2) + \
            theme_bw() + \
            labs(title=f"#Cell: {cell_num}") + \
            theme(
                text = element_text(family="Arial", size=5),
                title = element_text(family="Arial", size=6),
                axis_text = element_text(color = "black"),
                legend_title = element_text(family="Arial", size=6),
                legend_text = element_text(family="Arial", size=5),
                panel_grid = element_blank()
            )
        ggsave(p, filename=os.path.join(f_dir, "Division.pdf"), width=9, height=8, limitsize=False, units="cm")
        
        cell_adj_df = pd.DataFrame(self.cell_cell_interaction_li, columns=["CellName1", "CellName2"])
        cell_adj_df.to_csv(os.path.join(f_dir, "CellAdjustInfo.tsv"), sep="\t", index_label=False, index=False)
        cell_adj_df[["CellName1", "CellName2"]] = cell_adj_df[["CellName1", "CellName2"]].astype("str")
        cell1_df = cell_df[["CellName", "x", "y"]]
        cell1_df.columns = ["CellName1", "x1", "y1"]
        cell2_df = cell_df[["CellName", "x", "y"]]
        cell2_df.columns = ["CellName2", "x2", "y2"]
        cell_adj_info = cell_adj_df.merge(cell1_df, on="CellName1", how="left")
        cell_adj_info = cell_adj_info.merge(cell2_df, on="CellName2", how="left")

        cell_shape_df_li = list()
        nucleus_shape_df_li = list()
        for cell in self.cell_li:
            cell_x, cell_y = cell.cell_shape.polygon.boundary.xy
            cell_shape_df_li.append(pd.DataFrame({
                        "CellName": cell.name,
                        "CellType": cell.cell_type,
                        "CellShape": cell.cell_shape,
                        "x": cell_x,
                        "y": cell_y
                    }))
            nucleus_x, nucleus_y = cell.nucleus_shape.polygon.boundary.xy
            nucleus_shape_df_li.append(
                pd.DataFrame({
                        "CellName": cell.name,
                        "CellType": cell.cell_type,
                        "CellShape": cell.cell_shape,
                        "x": nucleus_x,
                        "y": nucleus_y
                    })
            )
        cell_shape_df = pd.concat(cell_shape_df_li)
        nucleus_shape_df = pd.concat(nucleus_shape_df_li)
        p = ggplot() + \
            geom_polygon(data=cell_shape_df, mapping=aes(x="x", y="y", fill="CellType", group="CellName"), color="black", alpha=0.4, size=0.2) + \
            geom_polygon(data=nucleus_shape_df, mapping=aes(x="x", y="y", group="CellName"), color="black", fill=None, size=0.2) + \
            geom_segment(data=cell_adj_info, mapping=aes(x="x1", xend="x2", y="y1", yend="y2"), size=2, color="yellow", alpha=0.5) + \
            theme_bw() + \
            labs(title=f"#Cell: {cell_num}") + \
            theme(
                text = element_text(family="Arial", size=5),
                title = element_text(family="Arial", size=6),
                axis_text = element_text(color = "black"),
                legend_title = element_text(family="Arial", size=6),
                legend_text = element_text(family="Arial", size=5),
                panel_grid = element_blank()
            )
        ggsave(p, filename=os.path.join(f_dir, "CellShape.pdf"), width=51, height=50, limitsize=False, units="cm")

        self.RNA_point_df = self.RNA_point_df[(self.RNA_point_df["x"]>=0) & (self.RNA_point_df["x"]<=self.map_info.x_range)]
        self.RNA_point_df = self.RNA_point_df[(self.RNA_point_df["y"]>=0) & (self.RNA_point_df["y"]<=self.map_info.y_range)]
        self.RNA_point_df.to_csv(os.path.join(f_dir, "RNA_points.tsv"), sep="\t", index_label=False, index=False)
