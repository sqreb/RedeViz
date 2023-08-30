import pandas as pd
import pickle
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import os

def extract_simulated_expr_main(args):
    f_emb_info = args.emb_info
    f_gene_li = args.gene_list
    f_spot_info = args.spot_info
    f_cell_shape = args.cell_shape
    f_out = args.output

    if not os.path.exists(f_out):
        os.makedirs(f_out)

    cell_shape_df = pd.read_csv(f_cell_shape, sep="\t")
    spot_info_df = pd.read_csv(f_spot_info, sep="\t")
    x_range = spot_info_df["spot_x_index"].max()
    y_range = spot_info_df["spot_y_index"].max()
    spot_info_df = spot_info_df[~np.isnan(spot_info_df["CellName"])]
    spot_info_df["CellName"] = spot_info_df["CellName"].astype(int)
    spot_info_df = spot_info_df.merge(cell_shape_df[["CellName", "ExpPointNum"]])
    cell_state_arr = coo_matrix((spot_info_df["BinIndex"].to_numpy().astype(int)+1, (spot_info_df["spot_x_index"].to_numpy(), spot_info_df["spot_y_index"].to_numpy())), [x_range+1, y_range+1]).toarray()
    cell_expr_arr = coo_matrix((spot_info_df["ExpPointNum"].to_numpy(), (spot_info_df["spot_x_index"].to_numpy(), spot_info_df["spot_y_index"].to_numpy())), [x_range+1, y_range+1]).toarray()

    with open(f_emb_info, "rb") as f:
        index = pickle.load(f)

    gene_bin_cnt = index["gene_bin_cnt"].to_dense().numpy()
    expand_gene_bin_cnt = np.zeros((gene_bin_cnt.shape[0]+1, gene_bin_cnt.shape[1]))
    expand_gene_bin_cnt[1:, :] = gene_bin_cnt
    expand_norm_gene_bin = expand_gene_bin_cnt / np.reshape(expand_gene_bin_cnt.sum(1), [-1, 1])
    expand_norm_gene_bin[0, :] = 0
    all_gene_li = np.array(index["gene_name"])

    if f_gene_li is None:
        gene_list = all_gene_li
    else:
        gene_list = list()
        with open(f_gene_li, "r") as f:
            for line in f.readlines():
                gid = line.rstrip("\n")
                if gid in all_gene_li:
                    gene_list.append(gid)

    for gid in gene_list:
        gid_index = int(np.where(all_gene_li==gid)[0])
        tmp_norm_gene_expr = expand_norm_gene_bin[:, gid_index]
        tmp_expr = tmp_norm_gene_expr[cell_state_arr] * cell_expr_arr
        save_npz(file=os.path.join(f_out, f"{gid}.simulated"), matrix=coo_matrix(tmp_expr))
