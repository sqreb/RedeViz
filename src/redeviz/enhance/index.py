import torch as tr
import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import pickle
from redeviz.enhance.utils import csr_mat2sparse_tensor, sparse_cnt_log_norm, sparse_cnt_mat2pct, norm_sparse_cnt_mat, select_device
import logging

def proprocess_SCE_HVG_embedding(sce: sc.AnnData, cell_type_label: str, embedding_resolution: float, no_HVG: bool, embedding="tSNE", embedding_dim=2, n_neighbors=15):
    sc.pp.normalize_total(sce, target_sum=1e4)
    sc.pp.log1p(sce)
    if no_HVG:
        HVG_sce = sce
    else:
        sc.pp.highly_variable_genes(sce, min_mean=0.0125, max_mean=3, min_disp=0.5)
        HVG_sce = sce[:, sce.var.highly_variable] 
    sc.pp.scale(HVG_sce)
    sc.tl.pca(HVG_sce, svd_solver='arpack')
    sc.tl.tsne(HVG_sce)
    sc.pp.neighbors(HVG_sce, n_neighbors=n_neighbors)
    sc.tl.umap(HVG_sce, n_components=embedding_dim)

    if embedding == "tSNE":
        cell_info_df = pd.DataFrame({
            "Index": np.arange(len(HVG_sce.obs)),
            "CellName": sce.obs_names,
            "CellType": HVG_sce.obs[cell_type_label],
            "Embedding1": HVG_sce.obsm["X_tsne"][:, 0],
            "Embedding2": HVG_sce.obsm["X_tsne"][:, 1],
            "Embedding3": 0
            })
    elif embedding == "UMAP":
        if embedding_dim == 2:
            cell_info_df = pd.DataFrame({
                "Index": np.arange(len(HVG_sce.obs)),
                "CellName": sce.obs_names,
                "CellType": HVG_sce.obs[cell_type_label],
                "Embedding1": HVG_sce.obsm["X_umap"][:, 0],
                "Embedding2": HVG_sce.obsm["X_umap"][:, 1],
                "Embedding3": 0
                })
        elif embedding_dim == 3:
            cell_info_df = pd.DataFrame({
                "Index": np.arange(len(HVG_sce.obs)),
                "CellName": sce.obs_names,
                "CellType": HVG_sce.obs[cell_type_label],
                "Embedding1": HVG_sce.obsm["X_umap"][:, 0],
                "Embedding2": HVG_sce.obsm["X_umap"][:, 1],
                "Embedding3": HVG_sce.obsm["X_umap"][:, 2]
                })
        else:
            raise ValueError()
    else:
        raise ValueError()
    cell_info_df["Embedding1"] -= cell_info_df["Embedding1"].min()
    cell_info_df["Embedding2"] -= cell_info_df["Embedding2"].min()
    cell_info_df["Embedding3"] -= cell_info_df["Embedding3"].min()

    cell_info_df["Embedding1"] = 200 * cell_info_df["Embedding1"] / cell_info_df["Embedding1"].max()
    cell_info_df["Embedding2"] = 200 * cell_info_df["Embedding2"] / cell_info_df["Embedding2"].max()
    if cell_info_df["Embedding3"].max() == 0:
        cell_info_df["Embedding3"] = 100
    else:
        cell_info_df["Embedding3"] = 200 * cell_info_df["Embedding3"] / cell_info_df["Embedding3"].max()
    cell_info_df["Embedding1BinIndex"] = np.floor(cell_info_df["Embedding1"] / embedding_resolution).astype(int)
    cell_info_df["Embedding2BinIndex"] = np.floor(cell_info_df["Embedding2"] / embedding_resolution).astype(int)
    cell_info_df["Embedding3BinIndex"] = np.floor(cell_info_df["Embedding3"] / embedding_resolution).astype(int)
    return cell_info_df


def compute_total_UMI_for_embedding_bins_worker(gene_cnt, select_cell_info_arr, bin_num, cell_num, gene_num, cell_info_num):
    select_cell_info_arr = tr.sparse_coo_tensor(tr.transpose(select_cell_info_arr, 1, 0), tr.ones(cell_info_num, dtype=tr.float32), [cell_num, bin_num])
    select_cell_info_arr = select_cell_info_arr.coalesce()
    t_select_cell_info_arr = select_cell_info_arr.transpose(1, 0)
    bin_gene_cnt = tr.matmul(t_select_cell_info_arr, gene_cnt.type(tr.float32))
    bin_gene_cnt = bin_gene_cnt.coalesce()
    return bin_gene_cnt


def compute_total_UMI_for_embedding_bins(gene_cnt: tr.Tensor, cell_info_df: pd.DataFrame, merge_method="None"):
    assert merge_method in ["None", "log"]
    select_cell_info_df = cell_info_df[~np.isnan(cell_info_df["BinIndex"])]
    select_cell_info_df["BinIndex"] = select_cell_info_df["BinIndex"].astype(np.int64)
    select_cell_info_arr = select_cell_info_df[["Index", "BinIndex"]].to_numpy()
    bin_num = select_cell_info_arr[:, 1].max() + 1
    cell_info_num = select_cell_info_arr.shape[0]
    select_cell_info_arr = tr.tensor(select_cell_info_arr)
    cell_num, gene_num = gene_cnt.shape
    if merge_method == "log":
        # log(TPM+1)
        gene_cnt = sparse_cnt_log_norm(gene_cnt)
    bin_gene_cnt = compute_total_UMI_for_embedding_bins_worker(gene_cnt, select_cell_info_arr, int(bin_num), int(cell_num), int(gene_num), int(cell_info_num))
    if merge_method == "log":
        val = tr.exp(bin_gene_cnt.values()) - 1
        val = tr.where(val<0, 0, val)
        bin_gene_cnt = tr.sparse_coo_tensor(bin_gene_cnt.indices(), val, bin_gene_cnt.shape)
    return bin_gene_cnt

def simulate_stereo_data_from_bin_scRNA_worker(cnt_arr, UMI_per_bin, mu_rand_arr, main_bin_type_ratio):
    cnt_arr = cnt_arr.to_dense()
    cnt_arr = tr.squeeze(cnt_arr)
    cnt_arr = cnt_arr + 1
    mu_bin_arr = UMI_per_bin * cnt_arr / tr.sum(cnt_arr)
    mu_bin_arr = tr.unsqueeze(mu_bin_arr, 0)
    ave_mu_bin_arr = mu_bin_arr * main_bin_type_ratio + mu_rand_arr * (1-main_bin_type_ratio)
    simu_arr = tr.distributions.poisson.Poisson(ave_mu_bin_arr).sample()
    simu_arr = simu_arr.to_sparse_coo()
    return simu_arr

def compute_neighbor_bin_loc(bin_info: pd.DataFrame, expand_size: float):
    ori_bin_index_li = list()
    neighbor_bin_index_li = list()

    for tmp_embedding1_bin_index in np.unique(bin_info["Embedding1BinIndex"]):
        tmp1_df = bin_info[bin_info["Embedding1BinIndex"]==tmp_embedding1_bin_index]
        tmp1_neighbor_df = bin_info[np.abs(bin_info["Embedding1BinIndex"].to_numpy()-tmp_embedding1_bin_index)<=expand_size]
        for tmp_embedding2_bin_index in np.unique(tmp1_df["Embedding2BinIndex"]):
            tmp2_df = tmp1_df[tmp1_df["Embedding2BinIndex"]==tmp_embedding2_bin_index]
            tmp2_neighbor_df = tmp1_neighbor_df[np.abs(tmp1_neighbor_df["Embedding2BinIndex"].to_numpy()-tmp_embedding2_bin_index)<=expand_size]
            for index in range(len(tmp2_df)):
                ori_bin_index = tmp2_df["BinIndex"].to_numpy()[index]
                ori_embedding3_bin_index = tmp2_df["Embedding3BinIndex"].to_numpy()[index]
                dist1 = tmp2_neighbor_df["Embedding1BinIndex"].to_numpy() - tmp_embedding1_bin_index
                dist2 = tmp2_neighbor_df["Embedding2BinIndex"].to_numpy() - tmp_embedding2_bin_index
                dist3 = tmp2_neighbor_df["Embedding3BinIndex"].to_numpy() - ori_embedding3_bin_index
                cond = np.sqrt(dist1**2 + dist2**2 + dist3**2) <= expand_size
                neighbor_bin_index = tmp2_neighbor_df[cond]["BinIndex"].to_numpy()
                ori_bin_index_li += [ori_bin_index] * len(neighbor_bin_index)
                neighbor_bin_index_li += list(neighbor_bin_index)
    bin_num = len(bin_info)
    loc_arr = tr.sparse_coo_tensor([ori_bin_index_li, neighbor_bin_index_li], np.ones(len(ori_bin_index_li)), [bin_num, bin_num])
    loc_arr = loc_arr.coalesce()
    loc_arr = loc_arr.to_dense()
    return loc_arr

def compute_simu_info_worker(cnt_arr, norm_method, UMI_per_bin, mu_rand_arr, main_bin_type_ratio, t_norm_gene_bin_cnt, quantile_index, neighbor_bin_expand_arr):
    assert norm_method in ["None", "log"]
    simu_cnt_arr = simulate_stereo_data_from_bin_scRNA_worker(cnt_arr, UMI_per_bin, mu_rand_arr, main_bin_type_ratio)
    if norm_method == "log":
        simu_cnt_arr = sparse_cnt_log_norm(simu_cnt_arr)
    norm_simu_cnt_arr = norm_sparse_cnt_mat(simu_cnt_arr)
    cos_simi = tr.matmul(norm_simu_cnt_arr.to_dense().type(tr.float32), t_norm_gene_bin_cnt.to_dense().type(tr.float32))
    max_cos_simi = tr.unsqueeze(tr.max(cos_simi, 1)[0], -1)
    sort_max_cos_simi, _ = tr.sort(tr.reshape(max_cos_simi, [-1]))
    quantile_cos_simi = tr.gather(sort_max_cos_simi, 0, quantile_index.type(tr.int64))
    is_max_cos_simi = cos_simi == max_cos_simi
    is_max_cos_simi = tr.unsqueeze(is_max_cos_simi, 0)
    is_max_cos_simi = is_max_cos_simi.type(tr.float64)
    neightbor_is_max_cos_simi = tr.matmul(is_max_cos_simi, neighbor_bin_expand_arr)
    neightbor_max_cos_simi_ratio = tr.mean(neightbor_is_max_cos_simi, 1)
    return quantile_cos_simi, neightbor_max_cos_simi_ratio

def compute_simu_info_random_worker(mu_rand_arr, norm_method, t_norm_gene_bin_cnt, quantile_index, neighbor_bin_expand_arr):
    assert norm_method in ["None", "log"]
    simu_cnt_arr = tr.distributions.poisson.Poisson(mu_rand_arr).sample()
    simu_cnt_arr = simu_cnt_arr.to_sparse_coo()
    if norm_method == "log":
        simu_cnt_arr = sparse_cnt_log_norm(simu_cnt_arr)
    norm_simu_cnt_arr = norm_sparse_cnt_mat(simu_cnt_arr)
    cos_simi = tr.matmul(norm_simu_cnt_arr.to_dense(), t_norm_gene_bin_cnt.to_dense())
    max_cos_simi = tr.unsqueeze(tr.max(cos_simi, 1)[0], -1)
    sort_max_cos_simi, _ = tr.sort(tr.reshape(max_cos_simi, [-1]))
    quantile_cos_simi = tr.gather(sort_max_cos_simi, 0, quantile_index.type(tr.int64))
    is_max_cos_simi = cos_simi == max_cos_simi
    is_max_cos_simi = tr.unsqueeze(is_max_cos_simi, 0)
    is_max_cos_simi = is_max_cos_simi.type(tr.float64)
    neightbor_is_max_cos_simi = tr.matmul(is_max_cos_simi, neighbor_bin_expand_arr)
    neightbor_max_cos_simi_ratio = tr.mean(neightbor_is_max_cos_simi, 1)
    return quantile_cos_simi, neightbor_max_cos_simi_ratio

def compute_mu_rand(simulate_num, embedding_bin_num, UMI_per_bin, gene_bin_cnt):
    rand_fac = tr.distributions.categorical.Categorical(logits=tr.log(tr.tensor([0.5, 0.2], device=gene_bin_cnt.device))).sample([simulate_num, embedding_bin_num])
    rand_fac = rand_fac.type(tr.float32)
    rand_cnt = tr.transpose(tr.matmul(tr.transpose(gene_bin_cnt, 1, 0), tr.transpose(rand_fac, 1, 0)), 1, 0)
    norm_fct = tr.unsqueeze(1/tr.sum(rand_cnt, 1), -1)
    mu_rand_arr = UMI_per_bin * rand_cnt * norm_fct
    return mu_rand_arr

def compute_simu_info(gene_bin_cnt, t_norm_gene_bin_cnt, norm_method, neighbor_bin_expand_arr, embedding_bin_num, gene_num, main_bin_type_ratio, UMI_per_bin, simulate_number_per_batch=1024, simulate_batch_num=20):
    device=gene_bin_cnt.device
    mu_rand_arr = compute_mu_rand(simulate_number_per_batch, embedding_bin_num, UMI_per_bin, gene_bin_cnt)

    quantile_index = tr.tensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], device=device) * (simulate_number_per_batch-1)
    quantile_index = quantile_index.type(tr.int32)
    quantile_cos_simi_li = list()
    neightbor_max_cos_simi_ratio_li = list()
    for bin_index in range(embedding_bin_num):
        if bin_index % 10 == 0:
            logging.info(f"{bin_index} / {embedding_bin_num}")
        tmp_cnt_arr = tr.index_select(gene_bin_cnt, 0, tr.tensor([bin_index], device=device))
        quantile_cos_simi = tr.zeros_like(quantile_index, dtype=tr.float32)
        neightbor_max_cos_simi_ratio = tr.zeros([neighbor_bin_expand_arr.shape[0], embedding_bin_num], dtype=tr.float64, device=device)
        for batch_index in range(simulate_batch_num):
            tmp_quantile_cos_simi, tmp_neightbor_max_cos_simi_ratio = compute_simu_info_worker(tmp_cnt_arr, norm_method, UMI_per_bin, mu_rand_arr, main_bin_type_ratio, t_norm_gene_bin_cnt, quantile_index, neighbor_bin_expand_arr)
            quantile_cos_simi = quantile_cos_simi + tmp_quantile_cos_simi
            neightbor_max_cos_simi_ratio = neightbor_max_cos_simi_ratio + tmp_neightbor_max_cos_simi_ratio
        quantile_cos_simi = quantile_cos_simi / simulate_batch_num
        neightbor_max_cos_simi_ratio = neightbor_max_cos_simi_ratio / simulate_batch_num
        quantile_cos_simi_li.append(quantile_cos_simi)
        neightbor_max_cos_simi_ratio_li.append(neightbor_max_cos_simi_ratio)
    
    rand_quantile_cos_simi = tr.zeros_like(quantile_index, dtype=tr.float32)
    rand_neightbor_max_cos_simi_ratio = tr.zeros([neighbor_bin_expand_arr.shape[0], embedding_bin_num], dtype=tr.float64, device=device)
    for batch_index in range(simulate_batch_num):
        tmp_rand_quantile_cos_simi, tmp_rand_neightbor_max_cos_simi_ratio = compute_simu_info_random_worker(mu_rand_arr, norm_method, t_norm_gene_bin_cnt, quantile_index, neighbor_bin_expand_arr)
        rand_quantile_cos_simi = rand_quantile_cos_simi + tmp_rand_quantile_cos_simi
        rand_neightbor_max_cos_simi_ratio = rand_neightbor_max_cos_simi_ratio + tmp_rand_neightbor_max_cos_simi_ratio
    rand_quantile_cos_simi = rand_quantile_cos_simi / simulate_batch_num
    rand_neightbor_max_cos_simi_ratio = rand_neightbor_max_cos_simi_ratio / simulate_batch_num
    quantile_cos_simi_li.append(rand_quantile_cos_simi)
    neightbor_max_cos_simi_ratio_li.append(rand_neightbor_max_cos_simi_ratio)
    quantile_cos_simi_arr = tr.stack(quantile_cos_simi_li, 0)
    neightbor_max_cos_simi_ratio_arr = tr.stack(neightbor_max_cos_simi_ratio_li, 1)
    return quantile_cos_simi_arr, neightbor_max_cos_simi_ratio_arr

def build_embedding_info_main(args):
    f_in = args.input
    f_out = args.output
    cell_type_label = args.cell_type_label
    gene_id_label = args.gene_id_label
    embedding_resolution = args.embedding_resolution
    embedding = args.embedding
    embedding_dim = args.embedding_dim
    n_neighbors = args.n_neighbors
    min_cell_num = args.min_cell_num
    f_gene_blacklist = args.gene_blacklist
    max_expr_ratio = args.max_expr_ratio
    cell_state_topN_gene = args.cell_state_topN_gene
    cell_state_topN_gene_max_ratio = args.cell_state_topN_gene_max_ratio
    no_HVG = args.no_HVG
    merge_method = args.merge_method

    assert embedding in ["tSNE", "UMAP"]
    if embedding == "tSNE":
        assert embedding_dim == 2
    else:
        assert embedding_dim in [2, 3]
    if embedding_resolution is None:
        if embedding == "tSNE":
            embedding_resolution = 2.5
        if embedding == "UMAP":
            if embedding_dim == 2:
                embedding_resolution = 2.5
            else:
                embedding_resolution = 6

    if not os.path.exists(f_out):
        os.makedirs(f_out)

    gene_blacklist = list()
    if f_gene_blacklist is not None:
        with open(f_gene_blacklist, "r") as f:
            for line in f.readlines():
                gene_blacklist.append(line.rstrip("\n"))
    
    logging.info(f"Loading single cell RNA-seq data from {f_in} ...")
    sce = sc.read_h5ad(f_in)

    # Remove genes in black list
    is_in_gene_blacklist = np.array([x in gene_blacklist for x in sce.var[gene_id_label]])
    sce = sce[:, ~is_in_gene_blacklist]

    # Remove high expressed genes
    gene_cnt = sce.X
    if not isinstance(gene_cnt, scipy.sparse._csr.csr_matrix):
        gene_cnt = scipy.sparse.csr_matrix(gene_cnt)

    gene_cnt = csr_mat2sparse_tensor(gene_cnt)
    gene_cnt = gene_cnt.type(tr.int64)
    total_UMI_per_gene = tr.sparse.sum(gene_cnt, 0)
    total_UMI_per_gene = total_UMI_per_gene.to_dense()
    gene_UMI_ratio = total_UMI_per_gene / tr.sum(total_UMI_per_gene)
    high_expr_gene = (gene_UMI_ratio > max_expr_ratio).numpy()
    
    sce = sce[:, ~high_expr_gene]
    gene_cnt = sce.X
    if not isinstance(gene_cnt, scipy.sparse._csr.csr_matrix):
        gene_cnt = scipy.sparse.csr_matrix(gene_cnt)
    gene_cnt = csr_mat2sparse_tensor(gene_cnt)
    gene_cnt = gene_cnt.type(tr.int64)
    gene_name_arr = list(sce.var[gene_id_label].to_numpy())

    logging.info(f"Computing embedding ...")
    cell_info_df = proprocess_SCE_HVG_embedding(sce, cell_type_label, embedding_resolution, no_HVG, embedding, embedding_dim, n_neighbors)

    embedding_info = cell_info_df.groupby(["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex"]).size().reset_index(name='CellNum')
    select_embedding_info = embedding_info[embedding_info["CellNum"]>=min_cell_num]
    select_embedding_info["BinIndex"] = np.arange(select_embedding_info.shape[0]).astype(int)
    select_embedding_info = select_embedding_info[["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex", "BinIndex"]]
    tmp_cell_info_df = cell_info_df.merge(select_embedding_info, on=["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex"], how="left")
    gene_bin_cnt = compute_total_UMI_for_embedding_bins(gene_cnt, tmp_cell_info_df, merge_method)
    gene_bin_pct = sparse_cnt_mat2pct(gene_bin_cnt)
    gene_bin_pct = gene_bin_pct.to_dense()
    sorted_gene_bin_pct, _ = tr.sort(gene_bin_pct, -1, descending=True)
    top_N_gene_pct = tr.sum(sorted_gene_bin_pct[:, :cell_state_topN_gene], -1)
    select_index = (top_N_gene_pct <= cell_state_topN_gene_max_ratio).numpy()
    select_embedding_info = select_embedding_info[select_index]
    select_embedding_info["BinIndex"] = np.arange(select_embedding_info.shape[0]).astype(int)
    cell_info_df = cell_info_df.merge(select_embedding_info, on=["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex"], how="left")

    total_cell_num = cell_info_df.shape[0]
    embedding_bin_num = select_embedding_info.shape[0]
    cell_num_not_in_embedding_bin = (np.isnan(cell_info_df["BinIndex"].to_numpy())).sum()
    logging.info(f"#Reference cells: {total_cell_num}\n#Embedding bins: {embedding_bin_num}\n#Cells out of embedding bins: {cell_num_not_in_embedding_bin}")
    cell_info_df.to_csv(os.path.join(f_out, "CellInfo.tsv"), sep="\t", index_label=False, index=False)

    cell_type_li = np.sort(np.unique(cell_info_df["CellType"].to_numpy()))
    cell_type_index_dict = {ct: index for (index, ct) in enumerate(cell_type_li)}
    bin_cell_type_num = cell_info_df.groupby(["BinIndex", "CellType"]).size().reset_index(name='CellNum')
    bin_cell_type_num["CellTypeIndex"] = np.array([cell_type_index_dict[ct] for ct in bin_cell_type_num["CellType"]])
    bin_cell_type_num_arr = tr.sparse_coo_tensor(np.transpose(bin_cell_type_num[["BinIndex", "CellTypeIndex"]].to_numpy(), [1, 0]), bin_cell_type_num["CellNum"].to_numpy(), [embedding_bin_num, len(cell_type_li)])
    bin_cell_type_num_arr = bin_cell_type_num_arr.coalesce()
    bin_cell_type_num_arr = bin_cell_type_num_arr.to_dense()
    bin_max_cell_type, bin_which_max_cell_type = tr.max(bin_cell_type_num_arr, 1)
    bin_max_cell_type_ratio = bin_max_cell_type / tr.sum(bin_cell_type_num_arr, 1)
    select_embedding_info["CellNum"] = tr.sum(bin_cell_type_num_arr, 1).numpy()
    select_embedding_info["MainCellType"] = cell_type_li[bin_which_max_cell_type.numpy()]
    select_embedding_info["MainCellTypeRatio"] = bin_max_cell_type_ratio.numpy()
    bin_max_cell_type_ratio_rm_main_label = bin_cell_type_num_arr.numpy().copy()
    for bin_index, cell_index in enumerate(bin_which_max_cell_type.numpy()):
        bin_max_cell_type_ratio_rm_main_label[bin_index, cell_index] = 0
    sec_max_val, sec_max_cell_type = tr.max(tr.tensor(bin_max_cell_type_ratio_rm_main_label), 1)
    sec_cell_type_ratio = sec_max_val / tr.sum(bin_cell_type_num_arr, 1)
    select_embedding_info["SecondaryCellType"] = cell_type_li[sec_max_cell_type.numpy()]
    select_embedding_info["SecondaryCellTypeRatio"] = sec_cell_type_ratio.numpy()
    select_embedding_info.to_csv(os.path.join(f_out, "Embedding.bin.info.tsv"), sep="\t", index_label=False, index=False)

    logging.info(f"Computing gene expression level for each embedding bin ...")
    gene_bin_cnt = compute_total_UMI_for_embedding_bins(gene_cnt, cell_info_df, merge_method)
    gene_bin_cnt = gene_bin_cnt.type(tr.float32)

    index_dict = {
        "gene_name": gene_name_arr,
        "gene_bin_cnt": gene_bin_cnt,
        "embedding": embedding,
        "embedding_resolution": embedding_resolution,
        "embedding_dim": embedding_dim,
        "cell_info": cell_info_df,
        "embedding_info": select_embedding_info,
    }
    f_index = os.path.join(f_out, "pretreat.pkl")
    with open(f_index, "wb") as f:
        pickle.dump(index_dict, f)

def build_embedding_index_main(args):
    f_in = args.input
    f_out = args.output
    bin_size_li = args.bin_size
    UMI_per_spot = args.UMI_per_spot
    norm_method = args.norm_method
    device = args.device_name

    with tr.no_grad():
        if device is None:
            device = select_device()
        main_bin_type_ratio_li = tr.tensor([0.75])

        with open(f_in, "rb") as f:
            embedding_dict = pickle.load(f)

        gene_name_arr = embedding_dict["gene_name"]
        gene_bin_cnt = embedding_dict["gene_bin_cnt"]
        embedding_resolution = embedding_dict["embedding_resolution"]
        neightbor_expand_size = np.arange(int(embedding_resolution*3))
        cell_info_df = embedding_dict["cell_info"]
        select_embedding_info = embedding_dict["embedding_info"]

        gene_num = len(gene_name_arr)
        gene_bin_cnt = gene_bin_cnt.to(device)
        if norm_method == "None":
            pass
        elif norm_method == "log":
            gene_bin_cnt = sparse_cnt_log_norm(gene_bin_cnt)
        else:
            raise ValueError("norm_method must in [None, log]")
        norm_gene_bin_cnt = norm_sparse_cnt_mat(gene_bin_cnt)

        t_norm_gene_bin_cnt = norm_gene_bin_cnt.transpose(1, 0)
        embedding_bin_num = select_embedding_info.shape[0]

        logging.info(f"Simulating spatial transcriptomics data and compute prior probability ...")
        neighbor_bin_expand_li = [compute_neighbor_bin_loc(select_embedding_info, expand_size) for expand_size in neightbor_expand_size]
        neighbor_bin_expand_arr = tr.stack(neighbor_bin_expand_li, 0)
        neighbor_bin_expand_arr = neighbor_bin_expand_arr.to(device)
        bin_quantile_cos_simi_li = list()
        bin_neightbor_max_cos_simi_ratio_li = list()
        for bin_size in bin_size_li:
            logging.info(f"Bin size: {bin_size}")
            UMI_per_bin = int(bin_size * bin_size * UMI_per_spot)
            main_type_ratio_quantile_cos_simi_li = list()
            main_type_ratio_neightbor_max_cos_simi_ratio_li = list()
            for main_bin_type_ratio in main_bin_type_ratio_li:
                logging.info(f"Main bin type cell ratio: {main_bin_type_ratio}")
                quantile_cos_simi_arr, neightbor_max_cos_simi_ratio_arr = compute_simu_info(gene_bin_cnt, t_norm_gene_bin_cnt, norm_method, neighbor_bin_expand_arr, embedding_bin_num, gene_num, main_bin_type_ratio, UMI_per_bin, simulate_number_per_batch=1024, simulate_batch_num=20)
                main_type_ratio_quantile_cos_simi_li.append(quantile_cos_simi_arr)
                main_type_ratio_neightbor_max_cos_simi_ratio_li.append(neightbor_max_cos_simi_ratio_arr)
            main_type_ratio_quantile_cos_simi_arr = tr.stack(main_type_ratio_quantile_cos_simi_li, 0)
            main_type_ratio_neightbor_max_cos_simi_ratio_arr = tr.stack(main_type_ratio_neightbor_max_cos_simi_ratio_li, 0)
            bin_quantile_cos_simi_li.append(main_type_ratio_quantile_cos_simi_arr)
            bin_neightbor_max_cos_simi_ratio_li.append(main_type_ratio_neightbor_max_cos_simi_ratio_arr)
        index_dict = {
            "index_type": "NGS",
            "norm_method": norm_method,
            "gene_name": gene_name_arr,
            "norm_embedding_bin_cnt": norm_gene_bin_cnt.to("cpu"),
            "neightbor_expand_size": neightbor_expand_size,
            "main_bin_type_ratio": main_bin_type_ratio_li,
            "bin_size": bin_size_li,
            "quantile_cos_simi": [x.to("cpu") for x in bin_quantile_cos_simi_li], # [bin_size, main_bin_type_ratio, SimuLabel: bin_size+1 (random), quantile_num]
            "neightbor_max_cos_simi_ratio": [x.to("cpu") for x in bin_neightbor_max_cos_simi_ratio_li],  # [bin_size, main_bin_type_ratio, neighbor_bin_expand_size, SimuLabel: bin_size (random), MaxSimiLabel: bin_size]
            "cell_info": cell_info_df,
            "embedding": embedding_dict["embedding"],
            "embedding_resolution": embedding_resolution,
            "embedding_dim": embedding_dict["embedding_dim"],
            "embedding_info": select_embedding_info
        }
        with open(f_out, "wb") as f:
            pickle.dump(index_dict, f)

