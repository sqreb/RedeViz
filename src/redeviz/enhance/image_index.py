import torch as tr
import torch.optim as optim
import numpy as np
import pickle
from redeviz.enhance.index import compute_neighbor_bin_loc
from redeviz.enhance.utils import sparse_cnt_mat2pct, select_device
import logging

def select_gene_bin_cnt_gene(gene_bin_cnt, gene_name_arr, select_gene_li, device):
    ## gene_bin_pct: <Bin, Gene>
    gene_bin_pct = sparse_cnt_mat2pct(gene_bin_cnt).to_dense()
    gene_bin_pct = gene_bin_pct.to(device)
    gene_name_arr = np.array(gene_name_arr)
    select_gene_index_arr = np.array([x in select_gene_li for x in gene_name_arr])
    gene_name_arr = gene_name_arr[select_gene_index_arr]
    dense_gene_bin_cnt = gene_bin_cnt.to_dense()[:,select_gene_index_arr]
    gene_bin_pct = gene_bin_pct[:,select_gene_index_arr]
    bin_num, gene_num = dense_gene_bin_cnt.shape
    gene_bin_pct_cutoff_li = list()
    for index in range(gene_num):
        tmp_gene_bin_pct = gene_bin_pct[:,index]
        if(tr.sum(tmp_gene_bin_pct)==0):
            gene_bin_pct_cutoff_li.append(0)
        else:
            gene_bin_pct_cutoff_li.append(tr.quantile(tmp_gene_bin_pct[tmp_gene_bin_pct>0], 0.95))
    gene_bin_pct_cutoff_arr = tr.tensor(gene_bin_pct_cutoff_li, device=device).reshape([1, -1])
    norm_gene_bin_pct = gene_bin_pct / gene_bin_pct_cutoff_arr
    norm_gene_bin_pct = tr.where(norm_gene_bin_pct>1, 1, norm_gene_bin_pct)
    norm_gene_bin_pct[:, gene_bin_pct_cutoff_arr.reshape([-1])==0] = 0
    return gene_bin_pct, norm_gene_bin_pct, gene_name_arr

def compute_image_ST_ref_dist(ST_bin_cnt, ST_bin_cnt_cutoff_arr, ref_norm_gene_bin_pct, fct):
    ## ST_bin_cnt: <1, X, Y, Gene>
    ## ST_bin_cnt_cutoff_arr: <Gene>
    ## ref_norm_gene_bin_pct: <Bin, Gene>
    ## Return: norm_ref_ST_dist <Bin, X, Y>
    ST_bin_cnt_cutoff_arr = ST_bin_cnt_cutoff_arr.reshape([1, 1, 1, -1])
    ref_norm_gene_bin_pct = ref_norm_gene_bin_pct.reshape([ref_norm_gene_bin_pct.shape[0], 1, 1, -1])
    norm_ST_bin_cnt = (ST_bin_cnt+1e-2) / (ST_bin_cnt_cutoff_arr+1e-2)
    norm_ST_bin_cnt = tr.where(norm_ST_bin_cnt>1, 1, norm_ST_bin_cnt)
    ref_ST_dist = tr.abs(tr.log2((ref_norm_gene_bin_pct+1e-3) / (norm_ST_bin_cnt+1e-3)))
    norm_ref_ST_dist = tr.sum(ref_ST_dist * fct.reshape([1, 1, 1, -1]), -1)
    return norm_ref_ST_dist

def infer_simulated_ST_bin_cnt_cutoff(gene_bin_pct, UMI_per_spot, bin_size, N_simu_sample = 1000):
    bin_num, gene_num = gene_bin_pct.shape
    pct_per_bin = tr.sum(gene_bin_pct, -1)
    ST_bin_cnt = tr.distributions.poisson.Poisson(gene_bin_pct*UMI_per_spot*bin_size*bin_size/tr.mean(pct_per_bin)).sample([N_simu_sample])

    ST_bin_cnt_cutoff_li = list()
    for index in range(gene_num):
        tmp_gene_bin_cnt = ST_bin_cnt[:,:,index]
        if(tr.sum(tmp_gene_bin_cnt)==0):
            ST_bin_cnt_cutoff_li.append(0)
        else:
            ST_bin_cnt_cutoff_li.append(tr.quantile(tmp_gene_bin_cnt[tmp_gene_bin_cnt>0], 0.95))
    ST_bin_cnt_cutoff_arr = tr.tensor(ST_bin_cnt_cutoff_li, device=gene_bin_pct.device)
    ST_bin_cnt.detach()
    if tr.cuda.is_available():
        tr.cuda.empty_cache()
    return ST_bin_cnt_cutoff_arr

def infer_weighted_fct(gene_bin_pct, ref_norm_gene_bin_pct, bin_pheno_dist, embedding_resolution, UMI_per_spot, bin_size):
    bin_num, gene_num = gene_bin_pct.shape
    device = gene_bin_pct.device
    ST_bin_cnt_cutoff_arr = infer_simulated_ST_bin_cnt_cutoff(gene_bin_pct, UMI_per_spot, bin_size)
    pheno_close_mask_arr = (bin_pheno_dist<(6/embedding_resolution)).type(tr.float32).to(device)
    pheno_far_mask_arr = (bin_pheno_dist>(12/embedding_resolution)).type(tr.float32).to(device)
    w = tr.ones([gene_num], device=device, requires_grad=True)
    optimizer = optim.Adam([w], 0.01)
    lass_loss = 1000
    best_epoch = 0
    best_fct = None
    mu_simu = gene_bin_pct*UMI_per_spot*bin_size*bin_size/tr.mean(tr.sum(gene_bin_pct, -1))
    for epoch in range(1000):
        optimizer.zero_grad()
        fct = tr.softmax(tr.exp(w), 0)
        ST_bin_cnt = tr.distributions.poisson.Poisson(mu_simu).sample([2]).unsqueeze(0)
        norm_ref_ST_dist = compute_image_ST_ref_dist(ST_bin_cnt, ST_bin_cnt_cutoff_arr, ref_norm_gene_bin_pct, fct)
        norm_ref_ST_dist = tr.where(norm_ref_ST_dist>4, 4, norm_ref_ST_dist)
        _, argmin_norm_ref_ST_dist = tr.min(norm_ref_ST_dist, 0)
        argmin_pheno_dist = bin_pheno_dist[tr.arange(bin_num), argmin_norm_ref_ST_dist]

        ave_norm_ref_ST_dist = norm_ref_ST_dist.mean(1)
        close_loss = tr.sum(ave_norm_ref_ST_dist * pheno_close_mask_arr) / tr.sum(pheno_close_mask_arr)
        far_loss = -1 * tr.sum(ave_norm_ref_ST_dist * pheno_far_mask_arr) / tr.sum(pheno_far_mask_arr)
        argmin_pheno_dist_loss1 = tr.mean((argmin_pheno_dist>12).type(tr.float32)).to(device)
        argmin_pheno_dist_loss2 = tr.mean((argmin_pheno_dist>9).type(tr.float32)).to(device)
        argmin_pheno_dist_loss3 = tr.mean((argmin_pheno_dist>6).type(tr.float32)).to(device)
        argmin_pheno_dist_loss4 = tr.mean((argmin_pheno_dist>0).type(tr.float32)).to(device)
        loss = close_loss + far_loss + argmin_pheno_dist_loss1 + argmin_pheno_dist_loss2 + argmin_pheno_dist_loss3 + argmin_pheno_dist_loss4 + 0.05 * fct.max() / fct.mean()
        loss.backward()
        optimizer.step()
        if lass_loss > loss:
            lass_loss = loss
            best_epoch = epoch
            best_fct = fct.detach()
            center_acc = 1-argmin_pheno_dist_loss4.detach().to("cpu").numpy()
            pheno6_acc = 1-argmin_pheno_dist_loss3.detach().to("cpu").numpy()
            pheno12_acc = 1-argmin_pheno_dist_loss1.detach().to("cpu").numpy()
            logging.info(f"""Epoch: {epoch} CenterAcc: {center_acc} PhenoDist6ACC: {pheno6_acc} PhenoDist12ACC: {pheno12_acc}""")
        else:
            if (epoch - best_epoch) > 5:
                break
    return best_fct, ST_bin_cnt_cutoff_arr

def simulate_img_ST_data_from_bin_scRNA_worker(mu_simu, mu_rand_arr, main_bin_type_ratio, N):
    simu_cnt_arr = tr.distributions.poisson.Poisson(mu_simu*main_bin_type_ratio + (1-main_bin_type_ratio) * mu_rand_arr).sample([N])
    return simu_cnt_arr

def compute_simu_info_worker(mu_simu, mu_rand_arr, simu_bin_index, fct, quantile_arr, ST_bin_cnt_cutoff_arr, ref_norm_gene_bin_pct, main_bin_type_ratio, neighbor_bin_expand_arr, N):
    bin_num = ref_norm_gene_bin_pct.shape[0]
    simu_cnt_arr = simulate_img_ST_data_from_bin_scRNA_worker(mu_simu, mu_rand_arr, main_bin_type_ratio, N)
    img_dist = compute_image_ST_ref_dist(simu_cnt_arr.reshape([1, 1, N, -1]), ST_bin_cnt_cutoff_arr, ref_norm_gene_bin_pct, fct)
    img_dist = img_dist.squeeze(1)
    img_simi = -1 * img_dist
    quantile_simi = tr.quantile(img_simi[simu_bin_index, :], quantile_arr, 0)
    max_simi, arg_max_simi = tr.max(img_simi, 0)
    arg_max_simi_ratio = tr.bincount(arg_max_simi, minlength=bin_num) / N
    neightbor_max_simi_ratio = tr.matmul(neighbor_bin_expand_arr.reshape([-1, bin_num]).type(tr.float32), arg_max_simi_ratio.reshape([-1, 1])).reshape([-1, bin_num])
    return quantile_simi, neightbor_max_simi_ratio


def compute_simu_info(gene_bin_pct, ref_norm_gene_bin_pct, neighbor_bin_expand_arr, ST_bin_cnt_cutoff_arr, fct, main_bin_type_ratio, UMI_per_bin, simulate_number_per_batch=1024, simulate_batch_num=20):
    device=gene_bin_pct.device

    bin_num, gene_num = gene_bin_pct.shape
    ave_gene_pct = gene_bin_pct.mean(0)
    mu_rand_arr = ave_gene_pct * UMI_per_bin / ave_gene_pct.sum()
    quantile_arr = tr.tensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1], device=device)
    quantile_simi_li = list()
    neightbor_max_simi_ratio_li = list()
    for bin_index in range(bin_num):
        if bin_index % 10 == 0:
            logging.info(f"{bin_index} / {bin_num}")
        tmp_pct_arr = gene_bin_pct[bin_index,]
        tmp_mu_simu_arr = tmp_pct_arr * UMI_per_bin / ave_gene_pct.sum()
        quantile_simi = tr.zeros_like(quantile_arr, dtype=tr.float32, device=device)
        neightbor_max_simi_ratio = tr.zeros([neighbor_bin_expand_arr.shape[0], bin_num], dtype=tr.float64, device=device)
        for batch_index in range(simulate_batch_num):
            tmp_quantile_simi, tmp_neightbor_max_simi_ratio = compute_simu_info_worker(tmp_mu_simu_arr, mu_rand_arr, bin_index, fct, quantile_arr, ST_bin_cnt_cutoff_arr, ref_norm_gene_bin_pct, main_bin_type_ratio, neighbor_bin_expand_arr, simulate_number_per_batch)
            quantile_simi += tmp_quantile_simi
            neightbor_max_simi_ratio += tmp_neightbor_max_simi_ratio
        quantile_simi = quantile_simi / simulate_batch_num
        neightbor_max_simi_ratio = neightbor_max_simi_ratio / simulate_batch_num
        quantile_simi_li.append(quantile_simi)
        neightbor_max_simi_ratio_li.append(neightbor_max_simi_ratio)
        

    rand_quantile_simi = tr.zeros_like(quantile_arr, dtype=tr.float32)
    rand_neightbor_max_simi_ratio = tr.zeros([neighbor_bin_expand_arr.shape[0], bin_num], dtype=tr.float64, device=device)
    for batch_index in range(simulate_batch_num):
        tmp_rand_quantile_simi, tmp_rand_neightbor_max_simi_ratio = compute_simu_info_worker(mu_rand_arr, mu_rand_arr, bin_index, fct, quantile_arr, ST_bin_cnt_cutoff_arr, ref_norm_gene_bin_pct, 1, neighbor_bin_expand_arr, simulate_number_per_batch)
        rand_quantile_simi = rand_quantile_simi + tmp_rand_quantile_simi
        rand_neightbor_max_simi_ratio = rand_neightbor_max_simi_ratio + tmp_rand_neightbor_max_simi_ratio
    rand_quantile_simi = rand_quantile_simi / simulate_batch_num
    rand_neightbor_max_simi_ratio = rand_neightbor_max_simi_ratio / simulate_batch_num
    quantile_simi_li.append(rand_quantile_simi)
    neightbor_max_simi_ratio_li.append(rand_neightbor_max_simi_ratio)
    quantile_simi_arr = tr.stack(quantile_simi_li, 0)
    neightbor_max_simi_ratio_arr = tr.stack(neightbor_max_simi_ratio_li, 1)
    return quantile_simi_arr, neightbor_max_simi_ratio_arr


def build_embedding_image_index_main(args):
    f_in = args.input
    f_out = args.output
    bin_size_li = args.bin_size  # (12, 21)
    UMI_per_spot = args.UMI_per_spot
    f_select_gene = args.select_gene
    device = args.device_name

    if device is None:
        device = select_device()
    main_bin_type_ratio_li = tr.tensor([0.75])

    with open(f_in, "rb") as f:
        embedding_dict = pickle.load(f)

    gene_name_arr = embedding_dict["gene_name"]
    gene_bin_cnt = embedding_dict["gene_bin_cnt"]
    embedding_resolution = embedding_dict["embedding_resolution"]
    neightbor_expand_size = np.arange(int(embedding_resolution*4))
    cell_info_df = embedding_dict["cell_info"]
    select_embedding_info = embedding_dict["embedding_info"]

    select_gene_li = list()
    with open(f_select_gene, "r") as f:
        for line in f.readlines():
            select_gene_li.append(line.rstrip("\n"))
    
    gene_bin_pct, ref_norm_gene_bin_pct, gene_name_arr = select_gene_bin_cnt_gene(gene_bin_cnt, gene_name_arr, select_gene_li, device)
    logging.info(f"{len(gene_name_arr)} genes were found in reference data")

    bin_emb_arr = tr.tensor(select_embedding_info[["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex"]].to_numpy(), dtype=tr.float32)
    bin_pheno_dist = tr.sqrt(tr.sum((bin_emb_arr.reshape([-1, 1, 3]) - bin_emb_arr.reshape([1, -1, 3])) ** 2 , -1)).to(device)

    logging.info(f"Computing gene normalize factor")
    gene_fct_li = list()
    ST_bin_cnt_cutoff_li = list()
    for bin_size in bin_size_li:
        logging.info(f"BinSize: {bin_size}")
        fct, ST_bin_cnt_cutoff_arr = infer_weighted_fct(gene_bin_pct, ref_norm_gene_bin_pct, bin_pheno_dist, embedding_resolution, UMI_per_spot, bin_size)
        gene_fct_li.append(fct)
        ST_bin_cnt_cutoff_li.append(ST_bin_cnt_cutoff_arr)
        if tr.cuda.is_available():
            tr.cuda.empty_cache()

    logging.info(f"Simulating spatial transcriptomics data and compute prior probability ...")
    neighbor_bin_expand_li = [compute_neighbor_bin_loc(select_embedding_info, expand_size) for expand_size in neightbor_expand_size]
    neighbor_bin_expand_arr = tr.stack(neighbor_bin_expand_li, 0)
    neighbor_bin_expand_arr = neighbor_bin_expand_arr.to(device)
    bin_quantile_simi_li = list()
    bin_neightbor_max_simi_ratio_li = list()
    for bin_size, fct, ST_bin_cnt_cutoff_arr in zip(bin_size_li, gene_fct_li, ST_bin_cnt_cutoff_li):
        logging.info(f"Bin size: {bin_size}")
        UMI_per_bin = int(bin_size * bin_size * UMI_per_spot)
        main_type_ratio_quantile_simi_li = list()
        main_type_ratio_neightbor_max_simi_ratio_li = list()
        for main_bin_type_ratio in main_bin_type_ratio_li:
            logging.info(f"Main bin type cell ratio: {main_bin_type_ratio}")
            quantile_simi_arr, neightbor_max_simi_ratio_arr = compute_simu_info(gene_bin_pct, ref_norm_gene_bin_pct, neighbor_bin_expand_arr, ST_bin_cnt_cutoff_arr, fct, main_bin_type_ratio, UMI_per_bin)
            main_type_ratio_quantile_simi_li.append(quantile_simi_arr)
            main_type_ratio_neightbor_max_simi_ratio_li.append(neightbor_max_simi_ratio_arr)
        main_type_ratio_quantile_simi_arr = tr.stack(main_type_ratio_quantile_simi_li, 0)
        main_type_ratio_neightbor_max_simi_ratio_arr = tr.stack(main_type_ratio_neightbor_max_simi_ratio_li, 0)
        bin_quantile_simi_li.append(main_type_ratio_quantile_simi_arr)
        bin_neightbor_max_simi_ratio_li.append(main_type_ratio_neightbor_max_simi_ratio_arr)
    index_dict = {
        "index_type": "Image",
        "gene_name": gene_name_arr,
        "norm_embedding_bin_pct": ref_norm_gene_bin_pct.to("cpu"),
        "gene_norm_fct": [x.to("cpu") for x in gene_fct_li],
        "neightbor_expand_size": neightbor_expand_size,
        "main_bin_type_ratio": main_bin_type_ratio_li,
        "bin_size": bin_size_li,
        "quantile_simi": [x.to("cpu") for x in bin_quantile_simi_li], # [bin_size, main_bin_type_ratio, SimuLabel: bin_size+1 (random), quantile_num]
        "neightbor_max_simi_ratio": [x.to("cpu") for x in bin_neightbor_max_simi_ratio_li],  # [bin_size, main_bin_type_ratio, neighbor_bin_expand_size, SimuLabel: bin_size (random), MaxSimiLabel: bin_size]
        "cell_info": cell_info_df,
        "embedding": embedding_dict["embedding"],
        "embedding_resolution": embedding_resolution,
        "embedding_dim": embedding_dict["embedding_dim"],
        "embedding_info": select_embedding_info
    }
    with open(f_out, "wb") as f:
        pickle.dump(index_dict, f)
