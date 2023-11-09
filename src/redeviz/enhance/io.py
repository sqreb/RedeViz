import torch as tr
import numpy as np
import pandas as pd
import pickle
from redeviz.enhance.utils import SparseSumPooling2DV2, norm_sparse_cnt_mat, sparse_cnt_log_norm, flat_coo_sparse_tensor_by_position
from redeviz.enhance.image_index import compute_image_ST_ref_dist


class RedeVizBinIndex(object):
    def __init__(self, f_pkl: str, cell_radius: int, device: str) -> None:
        with open(f_pkl, "rb") as f:
            dataset_dict = pickle.load(f)
        if "norm_method" in dataset_dict.keys():
            norm_method = dataset_dict["norm_method"]
        else:
            norm_method = "None"
        assert norm_method in ["None", "log"]
        self.device = device
        self.norm_method = norm_method
        self.bin_size = dataset_dict["bin_size"]
        self.bin_num = len(self.bin_size)
        self.max_bin_size = max(self.bin_size)
        self.cell_radius = cell_radius
        self.gene_name = dataset_dict["gene_name"]
        self.gene_num = len(self.gene_name)
        self.cell_info = dataset_dict["cell_info"]
        self.embedding_info = dataset_dict["embedding_info"]
        self.embedding_info["RowIndex"] = np.arange(self.embedding_info.shape[0])
        self.cell_info = self.cell_info[~np.isnan(self.cell_info["BinIndex"].to_numpy())]
        self.cell_info = self.embedding_info[["BinIndex", "RowIndex"]].merge(self.cell_info, how="inner")
        self.neightbor_expand_size = dataset_dict["neightbor_expand_size"]
        self.main_bin_type_ratio = dataset_dict["main_bin_type_ratio"]
        self.embedding_resolution = dataset_dict["embedding_resolution"]
        self.norm_embedding_bin_cnt = dataset_dict["norm_embedding_bin_cnt"]
        self.quantile_cos_simi = tr.stack(dataset_dict["quantile_cos_simi"], 0)
        self.quantile_cos_simi = tr.permute(self.quantile_cos_simi, [2, 0, 3, 1])  # EmbeddingBinIndex, BinSize, Pct, MixRatio
        self.quantile_cos_simi = tr.concat([self.quantile_cos_simi, tr.unsqueeze(self.quantile_cos_simi[-1, :, :, :], 0)], 0)
        self.neightbor_max_cos_simi_ratio = tr.stack(dataset_dict["neightbor_max_cos_simi_ratio"], 0)
        self.neightbor_max_cos_simi_ratio = tr.concat([self.neightbor_max_cos_simi_ratio, tr.unsqueeze(self.neightbor_max_cos_simi_ratio[:, :, :, -1, :], -2)], -2)
        self.neightbor_max_cos_simi_ratio = tr.permute(self.neightbor_max_cos_simi_ratio, [0, 1, 3, 4, 2])  # BinSize, MixRatio, EmbeddingBinLabel, MaxCosSimiEmbeddingBin, ExpandSize
        neightbor_max_cos_simi_max_score = tr.unsqueeze(tr.max(self.neightbor_max_cos_simi_ratio, -2)[0], -2)
        self.neightbor_max_cos_simi_score = self.neightbor_max_cos_simi_ratio / neightbor_max_cos_simi_max_score
        self.cell_type = np.sort(np.unique(self.cell_info["CellType"].to_numpy()))
        self.cell_type_num = len(self.cell_type)
        self.embedding_bin_num = self.embedding_info.shape[0]
        self.bin_cell_type_ratio = self.compute_bin_cell_type_ratio_arr(self.cell_type, self.cell_info, self.embedding_bin_num, self.cell_type_num).to(device)
        self.bin_dist = self.compute_bin_dist(self.embedding_info)
        self.quantile_cos_simi = self.quantile_cos_simi.to(device)
        self.neightbor_max_cos_simi_ratio = self.neightbor_max_cos_simi_ratio.to(device)
        self.neightbor_max_cos_simi_score = self.neightbor_max_cos_simi_score.to(device)
        self.norm_embedding_bin_cnt = self.norm_embedding_bin_cnt.to(device)
        self.bin_dist = self.bin_dist.to(device)


    def compute_bin_cell_type_ratio_arr(self, cell_type: np.array, cell_info: pd.DataFrame, embedding_bin_num: int, cell_type_num: int):
        cell_type_index_dict = {ct: index for (index, ct) in enumerate(cell_type)}
        bin_cell_type_num = cell_info.groupby(["RowIndex", "CellType"]).size().reset_index(name='CellNum')
        bin_cell_type_num = bin_cell_type_num[bin_cell_type_num["CellNum"]>0]
        bin_cell_type_num["CellTypeIndex"] = np.array([cell_type_index_dict[ct] for ct in bin_cell_type_num["CellType"].to_numpy()])
        bin_cell_type_num_arr = tr.sparse_coo_tensor(bin_cell_type_num[["RowIndex", "CellTypeIndex"]].to_numpy().T.astype(np.int64), bin_cell_type_num["CellNum"].to_numpy(), [embedding_bin_num, cell_type_num])
        bin_cell_type_num_arr = bin_cell_type_num_arr.coalesce()
        bin_cell_type_num_arr = bin_cell_type_num_arr.to_dense()
        bin_cell_type_ratio_arr = bin_cell_type_num_arr / tr.unsqueeze(tr.sum(bin_cell_type_num_arr, -1), -1)
        return bin_cell_type_ratio_arr

    def compute_bin_dist(self, embedding_info: pd.DataFrame):
        embedding1_bin_index_arr = tr.tensor(embedding_info["Embedding1BinIndex"].to_numpy()).type(tr.float32)
        embedding2_bin_index_arr = tr.tensor(embedding_info["Embedding2BinIndex"].to_numpy()).type(tr.float32)
        embedding3_bin_index_arr = tr.tensor(embedding_info["Embedding3BinIndex"].to_numpy()).type(tr.float32)
        embedding1_bin_index_dist = tr.abs(tr.unsqueeze(embedding1_bin_index_arr, 0) - tr.unsqueeze(embedding1_bin_index_arr, 1))
        embedding2_bin_index_dist = tr.abs(tr.unsqueeze(embedding2_bin_index_arr, 0) - tr.unsqueeze(embedding2_bin_index_arr, 1))
        embedding3_bin_index_dist = tr.abs(tr.unsqueeze(embedding3_bin_index_arr, 0) - tr.unsqueeze(embedding3_bin_index_arr, 1))
        bin_dist = tr.sqrt(embedding1_bin_index_dist**2 + embedding2_bin_index_dist**2 + embedding3_bin_index_dist**2)
        all_dist = np.ones([self.embedding_bin_num+2, self.embedding_bin_num+2]) * 200
        all_dist = all_dist - np.diag(np.diag(all_dist))
        all_dist[:self.embedding_bin_num, :self.embedding_bin_num] = bin_dist.numpy()
        res = tr.tensor(all_dist)
        return res

    def compute_cos_simi(self, spot_data, x_range, y_range, bin_size, block_size=2048):
        expand_size = int((bin_size - 1) / 2)
        spot_data = spot_data.type(tr.float32).to(self.norm_embedding_bin_cnt.device)
        if self.norm_method == "log":
            spot_data = sparse_cnt_log_norm(spot_data)
        smooth_spot_arr = SparseSumPooling2DV2(spot_data, expand_size, expand_size)
        norm_spot_arr = norm_sparse_cnt_mat(smooth_spot_arr)
        flat_norm_spot_arr = flat_coo_sparse_tensor_by_position(norm_spot_arr)
        flat_norm_spot_arr = tr.sparse_coo_tensor(flat_norm_spot_arr.indices()[1:,], flat_norm_spot_arr.values(), flat_norm_spot_arr.shape[1:])
        t_norm_bin_cnt = tr.transpose(self.norm_embedding_bin_cnt, 1, 0)
        t_norm_bin_cnt = t_norm_bin_cnt.coalesce()
        inner_product = tr.matmul(flat_norm_spot_arr, t_norm_bin_cnt.to_dense())
        inner_product = inner_product.to_dense()
        cos_simi = tr.reshape(inner_product, [1, x_range, y_range, self.embedding_bin_num])
        return cos_simi


class RedeVizImgNormBinIndex(RedeVizBinIndex):
    def __init__(self, f_pkl: str, cell_radius: int, device: str) -> None:
        with open(f_pkl, "rb") as f:
            dataset_dict = pickle.load(f)
        self.device = device
        self.bin_size = dataset_dict["bin_size"]
        self.bin_num = len(self.bin_size)
        self.max_bin_size = max(self.bin_size)
        self.cell_radius = cell_radius
        self.gene_name = dataset_dict["gene_name"]
        self.gene_num = len(self.gene_name)
        self.gene_norm_fct = [x.to(device) for x in dataset_dict["gene_norm_fct"]]
        self.cell_info = dataset_dict["cell_info"]
        self.embedding_info = dataset_dict["embedding_info"]
        self.embedding_info["RowIndex"] = np.arange(self.embedding_info.shape[0])
        self.cell_info = self.cell_info[~np.isnan(self.cell_info["BinIndex"].to_numpy())]
        self.cell_info = self.embedding_info[["BinIndex", "RowIndex"]].merge(self.cell_info, how="inner")
        self.neightbor_expand_size = dataset_dict["neightbor_expand_size"]
        self.main_bin_type_ratio = dataset_dict["main_bin_type_ratio"]
        self.embedding_resolution = dataset_dict["embedding_resolution"]
        self.norm_embedding_bin_pct = dataset_dict["norm_embedding_bin_pct"].to(device)
        self.quantile_cos_simi = tr.stack(dataset_dict["quantile_simi"], 0).to(device)
        self.quantile_cos_simi = tr.permute(self.quantile_cos_simi, [2, 0, 3, 1])  # EmbeddingBinIndex, BinSize, Pct, MixRatio
        self.quantile_cos_simi = tr.concat([self.quantile_cos_simi, tr.unsqueeze(self.quantile_cos_simi[-1, :, :, :], 0)], 0)
        self.neightbor_max_cos_simi_ratio = tr.stack(dataset_dict["neightbor_max_simi_ratio"], 0).to(device)
        self.neightbor_max_cos_simi_ratio = tr.concat([self.neightbor_max_cos_simi_ratio, tr.unsqueeze(self.neightbor_max_cos_simi_ratio[:, :, :, -1, :], -2)], -2)
        self.neightbor_max_cos_simi_ratio = tr.permute(self.neightbor_max_cos_simi_ratio, [0, 1, 3, 4, 2])  # BinSize, MixRatio, EmbeddingBinLabel, MaxCosSimiEmbeddingBin, ExpandSize
        neightbor_max_cos_simi_max_score = tr.unsqueeze(tr.max(self.neightbor_max_cos_simi_ratio, -2)[0], -2)
        self.neightbor_max_cos_simi_score = self.neightbor_max_cos_simi_ratio / neightbor_max_cos_simi_max_score
        self.cell_type = np.sort(np.unique(self.cell_info["CellType"].to_numpy()))
        self.cell_type_num = len(self.cell_type)
        self.embedding_bin_num = self.embedding_info.shape[0]
        self.bin_cell_type_ratio = self.compute_bin_cell_type_ratio_arr(self.cell_type, self.cell_info, self.embedding_bin_num, self.cell_type_num).to(device)
        self.bin_dist = self.compute_bin_dist(self.embedding_info).to(device)
        self.zero_bin_index = self.compute_zero_bin_index().to("cpu")
        self.ST_bin_cnt_cutoff_li = None

    def get_ST_norm_cutoff_worker(self, spot_data, bin_size, Q=0.95):
        _, x_range, y_range, gene_num = spot_data.shape
        spot_indices = spot_data.indices()
        spot_values = spot_data.values()
        bin_x_indices = (spot_indices[1,] / bin_size).type(tr.int64)
        bin_y_indices = (spot_indices[2,] / bin_size).type(tr.int64)
        bin_range = (tr.tensor(spot_data.shape) / bin_size).type(tr.int) + 1
        spot_bin_data = tr.sparse_coo_tensor(tr.stack([bin_x_indices, bin_y_indices, spot_indices[3,]], 0), spot_values, [bin_range[1], bin_range[2], gene_num])
        spot_bin_data = spot_bin_data.coalesce()
        bin_indices = spot_bin_data.indices()
        bin_gene_index = bin_indices[2,]
        bin_values = spot_bin_data.values()

        ST_bin_cnt_cutoff_li = list()
        for gene_index in range(gene_num):
            tmp_umi_arr = bin_values[bin_gene_index==gene_index]
            if tmp_umi_arr.sum() == 0:
                ST_bin_cnt_cutoff_li.append(0)
            else:
                ST_bin_cnt_cutoff_li.append(tr.quantile(tmp_umi_arr, Q))
        ST_bin_cnt_cutoff_arr = tr.tensor(ST_bin_cnt_cutoff_li).to(self.device)
        return ST_bin_cnt_cutoff_arr

    def get_ST_norm_cutoff(self, spot_data):
        ST_bin_cnt_cutoff_li = list()
        for bin_size in self.bin_size:
            ST_bin_cnt_cutoff_li.append(self.get_ST_norm_cutoff_worker(spot_data, bin_size))
        self.ST_bin_cnt_cutoff_li = ST_bin_cnt_cutoff_li

    def compute_cos_simi(self, spot_data):
        # simi_li: <1, X, Y, BinIndex, BinSize>
        # bg_simi_arr: <1, X, Y, 1, BinSize>
        spot_data = spot_data.coalesce()
        simi_li = list()
        bg_simi_li = list()
        zero_pct_arr = tr.zeros([1, self.gene_num], dtype=tr.float32, device=self.device)
        for bin_size, ST_bin_cnt_cutoff, gene_norm_fct in zip(self.bin_size, self.ST_bin_cnt_cutoff_li, self.gene_norm_fct):
            expand_size = int((bin_size - 1) / 2)
            spot_data = spot_data.type(tr.float32).to(self.norm_embedding_bin_pct.device)
            smooth_spot_arr = SparseSumPooling2DV2(spot_data, expand_size, expand_size)
            smooth_spot_arr = smooth_spot_arr.to_dense()
            
            norm_ref_ST_dist_li = list()
            bg_dist_li = list()
            for x_index in range(smooth_spot_arr.shape[1]):
                tmp_smooth_spot_arr = smooth_spot_arr[:, x_index, :, :].unsqueeze(1)
                tmp_norm_ref_ST_dist = compute_image_ST_ref_dist(tmp_smooth_spot_arr, ST_bin_cnt_cutoff, self.norm_embedding_bin_pct, gene_norm_fct)
                tmp_zero_ST_dist = compute_image_ST_ref_dist(tmp_smooth_spot_arr, ST_bin_cnt_cutoff, zero_pct_arr, gene_norm_fct)
                norm_ref_ST_dist_li.append(tmp_norm_ref_ST_dist)
                bg_dist_li.append(tmp_zero_ST_dist)
            norm_ref_ST_dist = tr.concat(norm_ref_ST_dist_li, 1)
            tmp_simi = -1 * tr.permute(norm_ref_ST_dist, [1, 2, 0])
            tmp_bg_simi = -1 * tr.permute(tr.concat(bg_dist_li, 1), [1, 2, 0])
            simi_li.append(tmp_simi)
            bg_simi_li.append(tmp_bg_simi)
        simi_arr = tr.stack(simi_li, -1).unsqueeze(0)
        bg_simi_arr = tr.stack(bg_simi_li, -1).unsqueeze(0)
        return simi_arr, bg_simi_arr

    def compute_zero_dist(self):
        ST_cnt_cutoff = tr.ones([self.gene_num], device=self.device)
        zero_dist_li = list()
        for gene_norm_fct in self.gene_norm_fct:
            zero_ST_cnt_arr = tr.zeros([1, 1, 1, self.gene_num], device=self.device)
            zero_dist = compute_image_ST_ref_dist(zero_ST_cnt_arr, ST_cnt_cutoff, self.norm_embedding_bin_pct, gene_norm_fct)
            zero_dist_li.append(zero_dist.reshape([-1]))
        return zero_dist_li

    def compute_zero_bin_index(self, topN=5):
        zero_dist_li = self.compute_zero_dist()
        ave_zero_dist_arr = tr.mean(tr.stack(zero_dist_li, -1), -1)
        s_zero_dist, zero_dist_order = tr.sort(ave_zero_dist_arr)
        zero_bin_index = zero_dist_order[:topN]
        return zero_bin_index


def load_spot_data(fname: str, x_label: str, y_label: str, gene_label: str, UMI_label: str, max_expr_ratio: float, dataset):
    df = pd.read_csv(fname, sep="\t")
    gene2index = {gid: index for index, gid in enumerate(dataset.gene_name)}
    x_li = df[x_label].to_numpy()
    y_li = df[y_label].to_numpy()
    gid_li = df[gene_label].to_numpy()
    UMI_li = df[UMI_label].to_numpy().astype(np.float32)
    x_range = x_li.max() + 1
    y_range = y_li.max() + 1

    indices_li = list()
    values_li = list()
    total_UMI_arr = np.zeros([x_range, y_range])
    gene_umi_arr = np.zeros_like(dataset.gene_name, dtype=np.float32)
    for (x, y, gid, UMI) in zip(x_li, y_li, gid_li, UMI_li):
        if UMI == 0:
            continue
        total_UMI_arr[x, y] += UMI
        if gid not in gene2index.keys():
            continue
        gene_index = gene2index[gid]
        indices_li.append((0, x, y, gene_index))
        values_li.append(UMI)
        gene_umi_arr[gene_index] += UMI

    gene_expr_ratio = gene_umi_arr / gene_umi_arr.sum()
    select_gene = gene_expr_ratio <= max_expr_ratio

    indices_li = np.array(indices_li)
    values_li = np.array(values_li)
    select_index = np.array([x for x in range(len(indices_li)) if select_gene[indices_li[x, 3]]])
    indices_li = indices_li[select_index,:]
    values_li = values_li[select_index]

    spot_data = tr.sparse_coo_tensor(indices_li.T.astype(np.int64), values_li, (1, x_range, y_range, dataset.gene_num))
    spot_data = spot_data.coalesce().type(tr.float32)
    total_UMI_arr = tr.reshape(tr.tensor(total_UMI_arr, dtype=tr.float32), (1, x_range, y_range, 1))
    return spot_data, total_UMI_arr, x_range, y_range

