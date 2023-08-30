import torch as tr
import numpy as np
import pandas as pd
import pickle
from redeviz.enhance.utils import SparseSumPooling2DV2, norm_sparse_cnt_mat, sparse_cnt_log_norm, flat_coo_sparse_tensor_by_position

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
        self.embedding_info = self.embedding_info.sort_values("BinIndex")
        self.embedding_info = self.embedding_info.reset_index()
        self.neightbor_expand_size = dataset_dict["neightbor_expand_size"]
        self.main_bin_type_ratio = dataset_dict["main_bin_type_ratio"]
        self.embedding_resolution = dataset_dict["embedding_resolution"]
        self.norm_embedding_bin_cnt = dataset_dict["norm_embedding_bin_cnt"].to(device)
        self.quantile_cos_simi = tr.stack(dataset_dict["quantile_cos_simi"], 0).to(device)
        self.quantile_cos_simi = tr.permute(self.quantile_cos_simi, [2, 0, 3, 1])  # EmbeddingBinIndex, BinSize, Pct, MixRatio
        self.quantile_cos_simi = tr.concat([self.quantile_cos_simi, tr.unsqueeze(self.quantile_cos_simi[-1, :, :, :], 0)], 0)
        self.neightbor_max_cos_simi_ratio = tr.stack(dataset_dict["neightbor_max_cos_simi_ratio"], 0).to(device)
        self.neightbor_max_cos_simi_ratio = tr.concat([self.neightbor_max_cos_simi_ratio, tr.unsqueeze(self.neightbor_max_cos_simi_ratio[:, :, :, -1, :], -2)], -2)
        self.neightbor_max_cos_simi_ratio = tr.permute(self.neightbor_max_cos_simi_ratio, [0, 1, 3, 4, 2])  # BinSize, MixRatio, EmbeddingBinLabel, MaxCosSimiEmbeddingBin, ExpandSize
        neightbor_max_cos_simi_max_score = tr.unsqueeze(tr.max(self.neightbor_max_cos_simi_ratio, -2)[0], -2)
        self.neightbor_max_cos_simi_score = self.neightbor_max_cos_simi_ratio / neightbor_max_cos_simi_max_score
        self.cell_type = np.sort(np.unique(self.cell_info["CellType"].to_numpy()))
        self.cell_type_num = len(self.cell_type)
        self.embedding_bin_num = self.embedding_info.shape[0]
        self.bin_cell_type_ratio = self.compute_bin_cell_type_ratio_arr(self.cell_type, self.cell_info, self.embedding_bin_num, self.cell_type_num).to(device)
        self.bin_dist = self.compute_bin_dist(self.embedding_info).to(device)

    def compute_bin_cell_type_ratio_arr(self, cell_type: np.array, cell_info: pd.DataFrame, embedding_bin_num: int, cell_type_num: int):
        cell_type_index_dict = {ct: index for (index, ct) in enumerate(cell_type)}
        bin_cell_type_num = cell_info.groupby(["BinIndex", "CellType"]).size().reset_index(name='CellNum')
        bin_cell_type_num["CellTypeIndex"] = np.array([cell_type_index_dict[ct] for ct in bin_cell_type_num["CellType"]])
        bin_cell_type_num_arr = tr.sparse_coo_tensor(bin_cell_type_num[["BinIndex", "CellTypeIndex"]].to_numpy().T.astype(np.int64), bin_cell_type_num["CellNum"].to_numpy(), [embedding_bin_num, cell_type_num])
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

