import anndata
import numpy as np
from numpy.random import poisson, choice
from scipy.stats import dirichlet
from redeviz.simulator.gene_expr import GeneExprInfo
from redeviz.simulator.cell import CellMap, CellMapInfo, DivInfo, LayoutInfo, ShapeInfo, PointNumInfo


class RefData(GeneExprInfo):
    def __init__(self, f_scRNA_h5ad: str,  cell_type_label: str, gene_id_label: str, NCR_label=None, min_cell_num=20):
        sce = anndata.read_h5ad(f_scRNA_h5ad)
        self.gene_li = sce.var[gene_id_label].to_numpy()
        if NCR_label is None:
            self.NCR_li = np.ones_like(self.gene_li)
        else:
            self.NCR_li = sce.var[NCR_label].to_numpy()
        cell_type_arr = sce.obs[cell_type_label].to_numpy()
        cell_type_li = sorted(set(cell_type_arr))
        self.cnt_mat_dict = dict()
        self.cell_num_dict = dict()
        self.total_expr_dict = dict()
        for cell_type in cell_type_li:
            cnt_mat = sce.X[cell_type_arr==cell_type,:]
            cell_num, gene_num = cnt_mat.shape
            if cell_num < min_cell_num:
                continue
            self.cnt_mat_dict[cell_type] = cnt_mat
            self.cell_num_dict[cell_type] = cell_num
            self.total_expr_dict[cell_type] = cnt_mat.sum() / cell_num
        self.cell_type_li = sorted(self.cnt_mat_dict.keys())
        self.ave_expr_li = np.array(sce.X.mean(0)).flatten()
    
    def simulate_gene_point_number(self, cell_type: str, N: float):
        cnt_mat = self.cnt_mat_dict[cell_type]
        cell_num = cnt_mat.shape[0]
        ave_expr = np.mean(cnt_mat[choice(range(cell_num), 5), :], 0)
        mu = np.array(N * ave_expr / ave_expr.sum()).flatten()
        gene_point_num = poisson(mu)
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


class scRnaCellMapInfo(CellMapInfo):
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
        exp_cell_num = x_range * y_range / 4e8
        scRNA_cell_num = sum(data_set.cell_num_dict.values())
        cell_num_fct = exp_cell_num / scRNA_cell_num
        for cell_type in self.cell_type:
            cell_num = data_set.cell_num_dict[cell_type]
            if (cell_num*cell_num_fct) > 40:
                div_cfg = {"div_type": "Normal", "div_num": int(cell_num*cell_num_fct)//20}
            else:
                div_cfg = {"div_type": "SingleCell", "div_num": int(np.ceil(cell_num*cell_num_fct)) }
            self.div_info[cell_type] = DivInfo(cell_type, div_cfg)
            self.layout_info[cell_type] = LayoutInfo(cell_type, {"layout_type": "Triangle"}, self.x_range, self.y_range, self.nucleis_radius[1])
            self.shape_info[cell_type] = ShapeInfo(cell_type, {"shape_type": "Random"})
            self.point_num_info[cell_type] = PointNumInfo(cell_type, {"RNA_per_cell_mu": 3000})


def simulate_by_scRNA_main(args):
    dataset = RefData(args.input, args.cell_type_label, args.gene_id_label, args.NCR_label, args.min_cell_num)
    cell_map_info = scRnaCellMapInfo(dataset, args.x_range, args.y_range, args.nucleis_radius, args.nucleus_shift_factor, args.uniform_noise_expr_ratio, args.diffusion_noise_expr_ratio, args.nucleus_RNA_capture_efficiency)
    cell_map = CellMap(cell_map_info)
    cell_map.simulate_all()
    cell_map.write_dataset(args.output)
