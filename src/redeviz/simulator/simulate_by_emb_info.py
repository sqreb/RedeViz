import numpy as np
import random
import logging
import pandas as pd
from plotnine import *
import os
from numpy.random import poisson
from scipy.stats import dirichlet
from collections import defaultdict
import pickle
import torch as tr
from redeviz.simulator.gene_expr import GeneExprInfo
from redeviz.simulator.cell import Cell, CellMap, CellMapInfo, DivInfo, LayoutInfo, ShapeInfo, PointNumInfo


class RefData(GeneExprInfo):
    def __init__(self, f_scRNA_pkl: str, NCR_label=None):
        with open(f_scRNA_pkl, "rb") as f:
            index = pickle.load(f)
        self.gene_li = np.array(index["gene_name"])
        self.gene_num = len(self.gene_li)
        if NCR_label is None:
            self.NCR_li = np.ones_like(self.gene_li, dtype=np.float32)
        else:
            self.NCR_li = np.array(index[NCR_label])
        emb_info = index["embedding_info"]
        emb_info = emb_info[emb_info["MainCellTypeRatio"]>0.6]
        cell_num_df = emb_info.groupby("MainCellType").agg({"CellNum":np.sum})
        self.cell_num_dict = {cell_type: num for (cell_type, num) in zip(cell_num_df.index, cell_num_df["CellNum"])}
        self.cell_type_bin_index_dict = defaultdict(list)
        for cell_type, bin_index in zip(emb_info["MainCellType"].to_numpy(), emb_info["BinIndex"].to_numpy()):
            self.cell_type_bin_index_dict[str(cell_type)].append(int(bin_index))
        self.gene_cnt = index["gene_bin_cnt"]
        self.cell_type_li = sorted(set(emb_info["MainCellType"]))
        self.ave_expr_li = tr.sparse.sum(self.gene_cnt, 0).to_dense().numpy().flatten()
        self.ave_expr_li = 1e5 * self.ave_expr_li / self.ave_expr_li.sum()
    
    def simulate_gene_point_number(self, emb_index: int, N: float):
        gene_cnt_arr = self.gene_cnt.to_dense()[emb_index, :]
        mu = N * gene_cnt_arr / tr.sum(gene_cnt_arr)
        gene_point_num = tr.distributions.poisson.Poisson(mu).sample([1])
        gene_point_num = tr.reshape(gene_point_num, [-1]).numpy()
        res_li = list()
        for gid, num, NCR in zip(self.gene_li, gene_point_num, self.NCR_li):
            if num == 0:
                continue
            res_li.append((gid, num, NCR))
        return res_li
    
    def simulate_ave_point_number(self, N: float):
        mu_li = N * dirichlet.rvs(self.ave_expr_li+1)[0]
        gene_point_num = poisson(mu_li)
        res_li = list()
        for gid, num, NCR in zip(self.gene_li, gene_point_num, self.NCR_li):
            if num == 0:
                continue
            res_li.append((gid, num, NCR))
        return res_li

class EmbBinCell(Cell):
    def __init__(self, name: str, x: float, y: float, cell_type: str, bin_index: int, shape_info: ShapeInfo, point_num_info: PointNumInfo):
        super().__init__(name, x, y, cell_type, shape_info, point_num_info)
        self.bin_index = bin_index
    
    def simulate_gene_point_number(self, gene_expr_info: RefData, N: int):
        return gene_expr_info.simulate_gene_point_number(self.bin_index, N)

class EmbBinCellMapInfo(CellMapInfo):
    def __init__(self, data_set: RefData, x_range=3e6, y_range=3e6, nucleis_radius=(1500, 3000), nucleus_shift_factor=0.2, uniform_noise_expr_ratio=0.0001, diffusion_noise_expr_ratio=0.01, nucleus_RNA_capture_efficiency=1.0) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.nucleis_radius = nucleis_radius
        self.nucleus_shift_factor = nucleus_shift_factor
        self.nucleus_RNA_capture_efficiency = nucleus_RNA_capture_efficiency
        self.uniform_noise_expr_ratio = uniform_noise_expr_ratio
        self.diffusion_noise_expr_ratio = diffusion_noise_expr_ratio
        self.cell_type = list(data_set.cell_type_li)
        self.expr_info = data_set
        self.div_info = dict()
        self.layout_info = dict()
        self.shape_info = dict()
        self.point_num_info = dict()
        exp_cell_num = x_range * y_range / 2e8
        scRNA_cell_num = sum(data_set.cell_num_dict.values())
        cell_num_fct = exp_cell_num / scRNA_cell_num
        all_div_num = 0
        for cell_type in self.cell_type:
            cell_num = data_set.cell_num_dict[cell_type]
            exp_cell_num = int(np.ceil(cell_num*cell_num_fct))
            if (exp_cell_num>50) and ((cell_num/scRNA_cell_num) > 0.02):
                div_num = int(cell_num*cell_num_fct)//40
                div_cfg = {"div_type": "Normal", "div_num": div_num}
                all_div_num += div_num
            else:
                div_cfg = {"div_type": "SingleCell", "div_num": exp_cell_num}
            self.div_info[cell_type] = DivInfo(cell_type, div_cfg)
            self.layout_info[cell_type] = LayoutInfo(cell_type, {"layout_type": "Triangle"}, self.x_range, self.y_range, self.nucleis_radius[1])
            self.shape_info[cell_type] = ShapeInfo(cell_type, {"shape_type": "Random"})
            self.point_num_info[cell_type] = PointNumInfo(cell_type, {"RNA_per_cell_mu": 3000})
        if all_div_num < 4:
            raise ValueError("The field of view size is too small, please increase x-range and y-range.")

class EmbBinCellMap(CellMap):
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
                bin_index_li = np.random.choice(self.map_info.expr_info.cell_type_bin_index_dict[cell_type], len(layout_info.layout.cell_map.geoms), replace=True)
                cell_li += [EmbBinCell(cell_index+index, point.x, point.y, cell_type, bin_index, shape_info, point_num_info) for ((index, point), bin_index) in zip(enumerate(layout_info.layout.cell_map.geoms), bin_index_li)]
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
                    bin_index = np.random.choice(self.map_info.expr_info.cell_type_bin_index_dict[cell_type], 1, replace=True)
                    cell_li[index].bin_index = bin_index[0]
                cell_index += div_info.div_num
            cell_li = self.adjust_cell_coords(cell_li)
            self.cell_li = cell_li
            is_fail = False
            return is_fail
        is_fail = True
        return is_fail

    def write_dataset(self, f_dir):
        logging.info("Writting spatial dataset ...")
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        cell_num = len(self.cell_li)
        x_li = list()
        y_li = list()
        cell_type_li = list()
        bin_index_li = list()
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
            bin_index_li.append(cell.bin_index)
            mu_point_num_li.append(cell.mu_RNA_point)
        cell_df = pd.DataFrame({
            "CellName": cell_name_li,
            "CellType": cell_type_li,
            "BinIndex": bin_index_li,
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


def simulate_by_emb_info_main(args):
    dataset = RefData(args.input, args.NCR_label)
    cell_map_info = EmbBinCellMapInfo(dataset, args.x_range, args.y_range, args.nucleis_radius, args.nucleus_shift_factor, args.uniform_noise_expr_ratio, args.diffusion_noise_expr_ratio, args.nucleus_RNA_capture_efficiency)
    cell_map = EmbBinCellMap(cell_map_info)
    cell_map.simulate_all()
    cell_map.write_dataset(args.output)
