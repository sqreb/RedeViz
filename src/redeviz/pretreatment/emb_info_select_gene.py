import numpy as np
import torch as tr
import pickle


def emb_info_select_gene_main(args):
    f_in = args.input
    f_gene_li = args.gene_li
    f_out = args.output
    min_gene_num = args.min_gene_num
    min_UMI_num = args.min_UMI_num

    with open(f_in, "rb") as f:
        embedding_dict = pickle.load(f)

    select_gene_li = list()
    with open(f_gene_li, "r") as f:
        for line in f.readlines():
            select_gene_li.append(line.rstrip("\n"))
    
    ref_gene_name = embedding_dict["gene_name"]
    ref_bin_cnt = embedding_dict["gene_bin_cnt"]
    ref_embedding_info = embedding_dict["embedding_info"]

    select_gene_index = np.array([x in select_gene_li for x in ref_gene_name])
    new_ref_gene_name = np.array(ref_gene_name)[select_gene_index]
    new_ref_bin_cnt = ref_bin_cnt.to_dense()[:, select_gene_index]
    gene_per_bin = tr.sum(new_ref_bin_cnt>0, -1)
    UMI_per_bin = tr.sum(new_ref_bin_cnt, -1)
    select_bin_index = (gene_per_bin>=min_gene_num) & (UMI_per_bin>=min_UMI_num)
    new_ref_bin_cnt = new_ref_bin_cnt[select_bin_index,:]
    new_ref_embedding_info = ref_embedding_info[select_bin_index.numpy()]

    embedding_dict["gene_name"] = list(new_ref_gene_name)
    embedding_dict["gene_bin_cnt"] = new_ref_bin_cnt.to_sparse_coo()
    embedding_dict["embedding_info"] = new_ref_embedding_info

    with open(f_out, "wb") as f:
        pickle.dump(embedding_dict, f)