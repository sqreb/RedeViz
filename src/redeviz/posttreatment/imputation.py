import numpy as np
import os
import pandas as pd
import cv2 as cv
from scipy.sparse import coo_matrix, save_npz
import anndata
import pickle
from scipy.stats import norm
import logging
from redeviz.posttreatment.utils import filter_pred_df

def compute_spatial_score(sig_arr, sig_pos_arr, kernel):
    sig_arr_in_region = sig_arr[sig_pos_arr]
    norm_sig_arr = (sig_arr - np.mean(sig_arr_in_region)) / np.std(sig_arr_in_region)
    norm_sig_arr[~sig_pos_arr] = 0
    sm_norm_sig_arr = cv.filter2D(norm_sig_arr, -1, kernel)
    score_arr = norm_sig_arr * sm_norm_sig_arr
    score = np.mean(score_arr[sig_pos_arr])
    return score

def imputation_build_index_main(args):
    f_index = args.index
    f_sce = args.sce
    f_out = args.output
    gene_name_label = args.gene_name_label
    embedding_smooth_sigma = args.embedding_smooth_sigma

    with open(f_index, "rb") as f:
        embedding_info_dict = pickle.load(f)

    if embedding_smooth_sigma is None:
        embedding_smooth_sigma = 3

    embedding_cell_info = embedding_info_dict["cell_info"]
    sce = anndata.read_h5ad(f_sce)
    sce_index_df = pd.DataFrame({
        "CellName": sce.obs_names,
        "SceIndex": np.arange(len(sce.obs)),
    })
    embedding_cell_info = embedding_cell_info.merge(sce_index_df)
    sce = sce[embedding_cell_info["SceIndex"].to_numpy(),:]

    cnt_mat = sce.X
    if not isinstance(cnt_mat, np.ndarray):
        cnt_mat = cnt_mat.toarray()
    norm_cnt_mat = 1e3 * cnt_mat / cnt_mat.sum(-1).reshape([-1, 1])
    embedding_cell_mat = embedding_cell_info[["Embedding1", "Embedding2", "Embedding3"]].to_numpy()

    embedding_info = embedding_info_dict["embedding_info"]
    ref_embedding_mat = embedding_info[["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex"]].to_numpy() * embedding_info_dict["embedding_resolution"]
    embedding1_diff_mat = embedding_cell_mat[:, 0].reshape([1, -1]) - ref_embedding_mat[:, 0].reshape([-1, 1])
    embedding2_diff_mat = embedding_cell_mat[:, 1].reshape([1, -1]) - ref_embedding_mat[:, 1].reshape([-1, 1])
    embedding3_diff_mat = embedding_cell_mat[:, 2].reshape([1, -1]) - ref_embedding_mat[:, 2].reshape([-1, 1])
    embedding_dist_mat = np.sqrt(embedding1_diff_mat**2+embedding2_diff_mat**2+embedding3_diff_mat**2)
    embedding_weight_mat = norm.pdf(embedding_dist_mat/embedding_smooth_sigma)
    norm_embedding_weight_mat = embedding_weight_mat / embedding_weight_mat.sum(-1).reshape([-1, 1])
    smooth_embedding_cnt_mat = np.matmul(norm_embedding_weight_mat, norm_cnt_mat)
    smooth_embedding_dict = {
        "embedding_mat": ref_embedding_mat,
        "smooth_embedding_cnt": smooth_embedding_cnt_mat,
        "gene_li": sce.var[gene_name_label].to_numpy()
    }
    with open(f_out, "wb") as f:
        pickle.dump(smooth_embedding_dict, f)

def imputation_main(args):
    f_in = args.input
    f_spot = args.spot
    f_index = args.index
    f_gene_li = args.gene_list
    f_out = args.output
    spot_pos_pos_label_li = args.spot_pos_pos_label
    spot_UMI_label = args.spot_UMI_label
    embedding_smooth_sigma = args.embedding_smooth_sigma
    UMI_smooth_sigma = args.UMI_smooth_sigma
    keep_other = args.keep_other

    if not os.path.exists(f_out):
        os.makedirs(f_out)

    with open(f_index, "rb") as f:
        smooth_embedding_dict = pickle.load(f)
    embedding_index_mat = smooth_embedding_dict["embedding_mat"]
    embedding_index_cnt_mat = smooth_embedding_dict["smooth_embedding_cnt"]
    embedding_index_gene_li = smooth_embedding_dict["gene_li"]

    if f_gene_li is None:
        imputate_gene_list = embedding_index_gene_li
    else:
        imputate_gene_list = list()
        with open(f_gene_li, "r") as f:
            for line in f.readlines():
                gid = line.rstrip("\n")
                if gid in embedding_index_gene_li:
                    imputate_gene_list.append(gid)

    spot_df = pd.read_csv(f_spot, sep="\t")

    total_UMI_df = spot_df.groupby(spot_pos_pos_label_li)[spot_UMI_label].agg(np.sum)
    del spot_df
    total_UMI_df = total_UMI_df.reset_index()
    x_range = total_UMI_df[spot_pos_pos_label_li[0]].max() + 1
    y_range = total_UMI_df[spot_pos_pos_label_li[1]].max() + 1
    UMI_mat = coo_matrix((total_UMI_df[spot_UMI_label].to_numpy(), (total_UMI_df[spot_pos_pos_label_li[0]].to_numpy(), total_UMI_df[spot_pos_pos_label_li[1]].to_numpy())), (x_range, y_range))
    UMI_mat = UMI_mat.toarray()
    UMI_mat = UMI_mat.astype(np.float32)

    UMI_knrnel_size = 2 * int(3 * UMI_smooth_sigma) + 1
    sm_UMI_mat = cv.GaussianBlur(UMI_mat, (UMI_knrnel_size, UMI_knrnel_size), UMI_smooth_sigma, UMI_smooth_sigma)
    nbg_mat = sm_UMI_mat[sm_UMI_mat>0]
    norm_sm_UMI_mat = sm_UMI_mat / np.median(nbg_mat)

    pred_df = pd.read_csv(f_in, sep="\t")
    pred_df = pred_df[pred_df["RefCellTypeScore"] > pred_df["BackgroundScore"]]
    nbg_pred_df = pred_df[pred_df["LabelTransfer"]!="Background"]
    if args.denoise:
        nbg_pred_df = filter_pred_df(nbg_pred_df, min_spot_in_region=args.min_spot_num)
    
    nbg_pred_df = nbg_pred_df[nbg_pred_df["ArgMaxCellType"]!="Other"]
    if not keep_other:
        nbg_pred_df = nbg_pred_df[nbg_pred_df["LabelTransfer"]!="Other"]
    embedding_index_df = pd.DataFrame(embedding_index_mat, columns=["Embedding1", "Embedding2", "Embedding3"])
    embedding_index_df["EmbeddingState"] = np.arange(embedding_index_mat.shape[0])
    nbg_pred_df = nbg_pred_df.merge(embedding_index_df)
    nbg_label_arr = coo_matrix(([True] * nbg_pred_df.shape[0], (nbg_pred_df["x"].to_numpy(), nbg_pred_df["y"].to_numpy())), (x_range, y_range))
    nbg_label_arr = nbg_label_arr.toarray()

    embedding_knrnel_size = 2 * max(1, int(2 * embedding_smooth_sigma)) + 1
    for index, imputate_gene in enumerate(imputate_gene_list):
        logging.info(f"{imputate_gene} ({index} / {len(imputate_gene_list)})")
        if os.path.exists(os.path.join(f_out, f"{imputate_gene}.imputate.npz")):
            continue
        assert imputate_gene in embedding_index_gene_li, imputate_gene
        tmp_imputate_cnt = embedding_index_cnt_mat[:, embedding_index_gene_li==imputate_gene].reshape(-1)
        tmp_spot_imputate_cnt = tmp_imputate_cnt[nbg_pred_df["EmbeddingState"].to_numpy()]
        if np.sum(tmp_spot_imputate_cnt) == 0:
            continue
        tmp_spot_imputate_mat = coo_matrix((tmp_spot_imputate_cnt, (nbg_pred_df["x"].to_numpy(), nbg_pred_df["y"].to_numpy())), (x_range, y_range))
        tmp_spot_imputate_mat = tmp_spot_imputate_mat.toarray().astype(np.float32)
        sm_tmp_spot_imputate_mat = cv.GaussianBlur(tmp_spot_imputate_mat, (embedding_knrnel_size, embedding_knrnel_size), embedding_smooth_sigma, embedding_smooth_sigma)
        weight_sm_tmp_spot_imputate_mat = sm_tmp_spot_imputate_mat * norm_sm_UMI_mat
        if np.sum(weight_sm_tmp_spot_imputate_mat) == 0:
            continue
        save_npz(file=os.path.join(f_out, f"{imputate_gene}.imputate"), matrix=coo_matrix(weight_sm_tmp_spot_imputate_mat))

