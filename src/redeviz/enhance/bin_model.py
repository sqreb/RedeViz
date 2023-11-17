import torch as tr
import numpy as np
from redeviz.enhance.io import RedeVizBinIndex
from redeviz.enhance.utils import sparse_onehot, SparseSumPooling2DV2, flat_coo_sparse_tensor_by_position
from kornia.filters import median_blur
import logging

class RedeVizBinModel(object):
    def __init__(self, dataset: RedeVizBinIndex, spot_data: tr.Tensor, total_UMI_arr: tr.Tensor) -> None:
        _, x_range, y_range, _ = spot_data.shape
        self.device = dataset.device
        self.x_range = x_range
        self.y_range = y_range
        self.dataset = dataset
        self.spot_data = spot_data
        self.cos_simi_arr = None
        self.max_cos_simi_arr = None
        self.argmax_cos_simi_arr = None
        self.max_bin_expand_size = int((self.dataset.max_bin_size - 1) / 2)
        self.ave_expr_arr = None
        self.neighbor_padding_arr = None
        self.label_arr = None
        self.score_arr = None
        self.bg_score_arr = None
        self.padding_mast_arr = None
        self.cos_init_state_arr = None
        self.x_shift_arr = None
        self.y_shift_arr = None
        self.cos_simi_range = tr.Size([self.x_range-2*self.max_bin_expand_size, self.y_range-2*self.max_bin_expand_size])
        self.total_UMI_arr = total_UMI_arr
        bin_pos_np_arr = self.dataset.embedding_info[["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex"]].to_numpy()
        max_emb = bin_pos_np_arr.max()
        self.bin_pos_arr = tr.tensor(list(bin_pos_np_arr) + [(-1, -1, -1), (max_emb+1, max_emb+1, max_emb+1)], device=self.device)

    def detach(self):
        arr_li = [
            self.cos_simi_arr, self.max_cos_simi_arr, self.argmax_cos_simi_arr, self.ave_expr_arr, self.neighbor_padding_arr, 
            self.label_arr, self.score_arr, self.bg_score_arr, self.padding_mast_arr, self.cos_init_state_arr,
            self.x_shift_arr, self.y_shift_arr, self.bin_pos_arr]
        for arr in arr_li:
            if isinstance(arr, tr.Tensor):
                arr.detach()

    def compute_cos_simi_worker(self):
        cos_simi_li = list()
        for bin_size in self.dataset.bin_size:
            delta_expand_size = self.max_bin_expand_size
            cos_simi = self.dataset.compute_cos_simi(self.spot_data, self.x_range, self.y_range, bin_size)
            cos_simi = cos_simi[:, delta_expand_size: (-1*delta_expand_size), delta_expand_size: (-1*delta_expand_size), :]
            cos_simi_li.append(cos_simi)
        cos_simi_arr = tr.stack(cos_simi_li, -1)
        return cos_simi_arr

    def compute_cos_simi(self) -> None:
        cos_simi_arr = self.compute_cos_simi_worker()
        self.cos_simi_arr = cos_simi_arr
        self.max_cos_simi_arr, self.argmax_cos_simi_arr = tr.max(cos_simi_arr, 3)

    def compute_average_signal_worker(self, expand_size=3):
        bin_size = 2 * expand_size + 1
        delta_size = self.max_bin_expand_size
        smooth_umi = tr.permute(tr.nn.AvgPool2d(kernel_size=(bin_size, bin_size), stride=(1, 1), padding=expand_size)(
            tr.permute(self.total_UMI_arr, [0, 3, 1, 2])
            ), [0, 2, 3, 1])
        smooth_umi = smooth_umi[:, delta_size:(-1*delta_size), delta_size:(-1*delta_size), :]
        return smooth_umi

    def compute_average_signal(self, expand_size=3) -> None:
        self.ave_expr_arr = self.compute_average_signal_worker(expand_size).to(self.device)

    def init_label_arr_worker(self, low_expr_cutoff=0.3):
        total_cos_simi_arr = tr.sum(self.cos_simi_arr, -1)
        ave_expr_arr = self.ave_expr_arr
        is_low_expr = ave_expr_arr<low_expr_cutoff
        min_cos_simi_cutoff = tr.min(self.dataset.quantile_cos_simi[:, :, 0, :], 2)[0]
        flat_argmax_cos_simi_arr = tr.reshape(self.argmax_cos_simi_arr, [(self.x_range-2*self.max_bin_expand_size) * (self.y_range-2*self.max_bin_expand_size), self.dataset.bin_num])
        sp_flat_argmax_cos_simi_arr = (flat_argmax_cos_simi_arr+1).to_sparse_coo()
        sp_flat_argmax_cos_simi_arr_indices = sp_flat_argmax_cos_simi_arr.indices()
        sp_flat_argmax_cos_simi_arr_values = sp_flat_argmax_cos_simi_arr.values() - 1
        min_cos_simi_indices = tr.stack([sp_flat_argmax_cos_simi_arr_values, sp_flat_argmax_cos_simi_arr_indices[1, :]], -1)
        min_cos_simi_values = min_cos_simi_cutoff[list(min_cos_simi_indices.T)]
        min_cos_simi_arr = tr.reshape(min_cos_simi_values, [1, self.x_range-2*self.max_bin_expand_size, self.y_range-2*self.max_bin_expand_size, self.dataset.bin_num])
        low_cos_simi_arr = tr.unsqueeze(tr.all(self.max_cos_simi_arr < min_cos_simi_arr, -1), -1)
        which_max_arr = tr.max(total_cos_simi_arr, -1)[1].unsqueeze(-1)
        which_max_arr[low_cos_simi_arr] = self.dataset.embedding_bin_num
        which_max_arr[is_low_expr] = self.dataset.embedding_bin_num + 1
        return which_max_arr
    
    def init_label_arr(self, low_expr_cutoff=0.3) -> None:
        self.label_arr = self.init_label_arr_worker(low_expr_cutoff)

    def compute_signal_cov_score(self, mid_signal_cutoff=1.0):
        self.bg_score_arr = (mid_signal_cutoff - self.ave_expr_arr) / mid_signal_cutoff

    def init_neighbor_padding_indices(self, max_dist_cutoff=5):
        self.neighbor_padding_arr = (self.dataset.bin_dist<=max_dist_cutoff).type(tr.int64)[:self.dataset.embedding_bin_num,:]

    def get_neighbor_label(self, all_state_arr_indices, label_arr, padding_mast_arr, x_shift_arr, y_shift_arr, expand_size):
        x_coord_arr = tr.unsqueeze(all_state_arr_indices[:, 0], -1)
        y_coord_arr = tr.unsqueeze(all_state_arr_indices[:, 1], -1)
        padding_label_arr = tr.permute(tr.nn.ZeroPad2d(expand_size)(tr.permute(label_arr, [0, 3, 1, 2])), [0, 2, 3, 1])
        padding_label_arr = padding_label_arr + padding_mast_arr
        padding_label_arr = padding_label_arr[0, :, :, 0]
        x_shift_arr = x_shift_arr + x_coord_arr
        y_shift_arr = y_shift_arr + y_coord_arr
        flat_x_shift_arr = tr.reshape(x_shift_arr, [-1])
        flat_y_shift_arr = tr.reshape(y_shift_arr, [-1])
        indices = tr.stack([flat_x_shift_arr, flat_y_shift_arr], -1)
        flat_neighbor_label = padding_label_arr[list(indices.T)]
        return flat_neighbor_label

    def init_close_embedding_bin_score_mask_arr(self, expand_size):
        padding_mast_arr = np.ones([1, self.cos_simi_range[0]+2*expand_size, self.cos_simi_range[1]+2*expand_size, 1]) * (self.dataset.embedding_bin_num + 1)
        padding_mast_arr[:, expand_size:(-1*expand_size), expand_size:(-1*expand_size), :] = 0
        self.padding_mast_arr = tr.tensor(padding_mast_arr, dtype=tr.int64, device=self.device)
        x_shift_li = list()
        y_shift_li = list()
        for x in range(2*expand_size+1):
            for y in range(2*expand_size+1):
                if (x == expand_size) and (y == expand_size):
                    continue
                x_shift_li.append(x)
                y_shift_li.append(y)
        self.x_shift_arr = tr.tensor([x_shift_li], dtype=tr.int64, device=self.device)
        self.y_shift_arr = tr.tensor([y_shift_li], dtype=tr.int64, device=self.device)

    def compute_close_embedding_bin_score_ori_label_arr(self, all_state_arr_indices: tr.Tensor, label_arr: tr.Tensor, ave_bin_dist_cutoff=5, expand_size=3):
        rep_num = (2*expand_size+1) ** 2 - 1
        flat_neighbor_label = self.get_neighbor_label(all_state_arr_indices, label_arr, self.padding_mast_arr, self.x_shift_arr, self.y_shift_arr, expand_size)
        mid_label_arr = all_state_arr_indices[:, 2]
        rep_mid_label_arr = tr.unsqueeze(mid_label_arr, -1).repeat([1, rep_num])
        flat_mid_label_arr = tr.reshape(rep_mid_label_arr, [-1])
        dist_indices = tr.stack([flat_mid_label_arr, flat_neighbor_label], -1)
        flat_dist = self.dataset.bin_dist[list(dist_indices.T)]
        bin_dist = tr.reshape(flat_dist, [-1, rep_num])
        is_close_arr = (bin_dist < ave_bin_dist_cutoff).type(tr.float32)
        score_arr = tr.mean(is_close_arr, -1)
        return score_arr

    def compute_is_in_ref_cell_score(self, all_state_mid_label_arr: tr.Tensor,  all_state_mid_cos_simi_arr: tr.Tensor, batch_effect_fct=0.8):
        cos_simi_cutoff = self.dataset.quantile_cos_simi[:, :, 1, 0] * batch_effect_fct
        cos_simi_region = self.dataset.quantile_cos_simi[:, :, -1, 0] - self.dataset.quantile_cos_simi[:, :, 0, 0]
        all_state_cos_simi_cutoff = cos_simi_cutoff[list(all_state_mid_label_arr.T)]
        all_state_cos_simi_region = cos_simi_region[list(all_state_mid_label_arr.T)]
        score = all_state_mid_cos_simi_arr - all_state_cos_simi_cutoff
        score = score / all_state_cos_simi_region
        is_not_bg = (all_state_mid_label_arr != (self.dataset.embedding_bin_num+1)).type(tr.float32)
        score = score * is_not_bg
        is_out_class = (all_state_mid_label_arr == self.dataset.embedding_bin_num).type(tr.float32)
        score = score * (is_out_class - 0.5) * -2
        score_arr = tr.mean(score, -1)
        return score_arr

    def compute_argmax_prob_score(self, all_state_mid_label_arr: tr.Tensor, all_state_mid_argmax_cos_simi_arr: tr.Tensor, max_dist_cutoff=5):
        score_li = list()
        for bin_index in range(self.dataset.bin_num):
            tmp_neightbor_max_cos_simi_score = self.dataset.neightbor_max_cos_simi_score[bin_index, 0, :, :, :(max_dist_cutoff+1)]
            tmp_all_state_mid_argmax_cos_simi_arr = tr.unsqueeze(all_state_mid_argmax_cos_simi_arr[:, bin_index], -1)
            indices_arr = tr.concat([all_state_mid_label_arr, tmp_all_state_mid_argmax_cos_simi_arr], -1)
            tmp_score = tmp_neightbor_max_cos_simi_score[list(indices_arr.T)]
            tmp_score = tr.mean(tmp_score, -1)
            score_li.append(tmp_score)
        score_arr = tr.stack(score_li, -1)
        score_arr = tr.mean(score_arr, -1)
        score_arr = score_arr.type(tr.float32)
        return score_arr

    def generate_all_state_by_cos_simi_worker(self, topN=7, use_other=False):
        cos_simi_arr = tr.reshape(self.cos_simi_arr, [self.cos_simi_range[0]*self.cos_simi_range[1], self.dataset.embedding_bin_num, self.dataset.bin_num])
        _, top_simi_index = tr.topk(tr.permute(cos_simi_arr, [0, 2, 1]), k=topN, dim=-1)
        top_simi_index = tr.reshape(top_simi_index, [self.cos_simi_range[0]*self.cos_simi_range[1], topN*self.dataset.bin_num])
        top_simi_index = top_simi_index.type(tr.int64)
        neighbor_simi_index = self.neighbor_padding_arr[top_simi_index]
        state_arr = tr.any(neighbor_simi_index>0, 1)
        if use_other:
            state_arr = state_arr | tr.reshape(tr.tensor([False]*self.dataset.embedding_bin_num+[True]*2, device=self.device), [1, -1])
        else:
            state_arr = state_arr | tr.reshape(tr.tensor([False]*(self.dataset.embedding_bin_num+1)+[True], device=self.device), [1, -1])
        return state_arr
    
    def generate_all_state_by_cos_simi(self, topN=7, use_other=False):
        self.cos_init_state_arr = self.generate_all_state_by_cos_simi_worker(topN, use_other)

    def generate_all_state_ori_label_arr(self, label_arr: tr.Tensor):
        flat_label_arr = tr.reshape(label_arr, [-1])
        flat_label_one_hot = tr.nn.functional.one_hot(flat_label_arr, num_classes=self.dataset.embedding_bin_num+2).type(tr.bool)
        state_arr = self.cos_init_state_arr | flat_label_one_hot
        state_arr = tr.reshape(state_arr, [self.cos_simi_range[0], self.cos_simi_range[1], self.dataset.embedding_bin_num+2])
        state_arr_indices = tr.stack(tr.where(state_arr), 0).T
        return state_arr_indices

    def evaluate_score_new(self, label_arr: tr.Tensor, neighbor_close_label_fct=0.2, signal_cov_score_fct=1.0, is_in_ref_score_fct=1.0, argmax_prob_score_fct=1.0, ave_bin_dist_cutoff=4, embedding_expand_size=3, batch_effect_fct=0.8):
        bg_score_arr = self.bg_score_arr
        sig_score_arr = -1 * bg_score_arr
        all_state_arr_indices = self.generate_all_state_ori_label_arr(label_arr)
        all_state_argmax_cos_simi_arr = self.argmax_cos_simi_arr[0, :, :, :][list(all_state_arr_indices[:, :2].T)]
        cos_simi_arr = self.cos_simi_arr[0, :, :, :]
        ave_cos_simi_arr = tr.unsqueeze(tr.mean(cos_simi_arr, 2), 2)
        expand_cos_simi_arr = tr.concat([cos_simi_arr, ave_cos_simi_arr, ave_cos_simi_arr], 2)
        all_state_cos_simi_arr = expand_cos_simi_arr[list(all_state_arr_indices.T)]
        all_state_label_arr = tr.unsqueeze(all_state_arr_indices[:, 2], -1)
        close_ratio_score = self.compute_close_embedding_bin_score_ori_label_arr(all_state_arr_indices, label_arr, ave_bin_dist_cutoff, embedding_expand_size)
        is_in_ref_cell_score = self.compute_is_in_ref_cell_score(all_state_label_arr, all_state_cos_simi_arr, batch_effect_fct)
        shift_score = tr.min(is_in_ref_cell_score)
        is_in_ref_cell_score = is_in_ref_cell_score - shift_score
        argmax_prob_score = self.compute_argmax_prob_score(all_state_label_arr, all_state_argmax_cos_simi_arr, ave_bin_dist_cutoff)
        sp_score_arr_value = neighbor_close_label_fct * close_ratio_score + is_in_ref_score_fct*is_in_ref_cell_score + argmax_prob_score_fct*argmax_prob_score
        sp_score_arr = tr.sparse_coo_tensor(all_state_arr_indices.T, sp_score_arr_value, [self.cos_simi_range[0], self.cos_simi_range[1], self.dataset.embedding_bin_num+2])
        score_arr = sp_score_arr.to_dense()
        score_arr = score_arr + shift_score
        score_arr = tr.unsqueeze(score_arr, 0)
        score_arr = score_arr + signal_cov_score_fct * tr.concat([sig_score_arr.repeat([1, 1, 1, self.dataset.embedding_bin_num+1]), bg_score_arr], -1)
        if tr.cuda.is_available():
            tr.cuda.empty_cache()
        return score_arr

    def compute_close_label_num_worker(self, tmp_flat_sum_label_num: tr.Tensor, close_dist_arr: tr.Tensor):      
        tmp_flat_sum_label_num = tmp_flat_sum_label_num.to_dense()[0, :, :]
        tmp_res = tr.sum(tr.matmul(tmp_flat_sum_label_num, close_dist_arr.squeeze(0)) * tmp_flat_sum_label_num, 1)
        return tmp_res

    def compute_close_label_num(self, flat_sum_label_num: tr.Tensor, close_dist_arr: tr.Tensor, block_size=128):
        res = self.compute_close_label_num_worker(flat_sum_label_num, close_dist_arr)
        return res

    def compute_adjust_index_worker(self, label_arr: tr.Tensor, expand_size=2, bin_dist_cutoff=3):
        close_dist_arr = (self.dataset.bin_dist <= bin_dist_cutoff).type(tr.float32)
        close_dist_arr = tr.unsqueeze(close_dist_arr, 0)
        one_hot_label = sparse_onehot(tr.squeeze(label_arr, -1), self.dataset.embedding_bin_num+2)
        pool_size = expand_size * 2 + 1
        sum_label_num = SparseSumPooling2DV2(one_hot_label, pool_size, pool_size)
        sum_label_num = sum_label_num.type(tr.float32)
        flat_sum_label_num = flat_coo_sparse_tensor_by_position(sum_label_num)
        total_label_fac = tr.sparse.sum(flat_sum_label_num, [0, 2]) ** 2
        total_label_fac = total_label_fac.to_dense()
        return flat_sum_label_num, close_dist_arr, total_label_fac

    def compute_adjust_index(self, label_arr: tr.Tensor, expand_size=2, bin_dist_cutoff=3, same_label_ratio=0.9):
        flat_sum_label_num, close_dist_arr, total_label_fac = self.compute_adjust_index_worker(label_arr, expand_size, bin_dist_cutoff)
        close_label_ratio = self.compute_close_label_num(flat_sum_label_num, close_dist_arr) / total_label_fac
        adjust_index_arr = tr.reshape(close_label_ratio > same_label_ratio, [1, self.cos_simi_range[0], self.cos_simi_range[1], 1])
        return adjust_index_arr

    def adjust_label_arr(self, label_arr: tr.Tensor, expand_size=2, bin_dist_cutoff=3, same_label_ratio=0.8):
        embedding_bin_num = tr.tensor(self.dataset.embedding_bin_num, dtype=label_arr.dtype, device=self.device)
        bg_pos_arr = label_arr == (embedding_bin_num + 1)
        other_pos_arr = label_arr == embedding_bin_num
        sig_pos_arr = label_arr < embedding_bin_num
        adj_label_arr = tr.where(bg_pos_arr, embedding_bin_num, label_arr)
        outer_pos_indices = tr.where(adj_label_arr==embedding_bin_num)
        outer_pos_indices = tr.stack(outer_pos_indices, 0).T
        flat_indices = tr.sum(outer_pos_indices, -1)
        is_odd = flat_indices % 2 == 1
        select_outer_pos_indices = outer_pos_indices[is_odd]
        select_outer_pos_arr = tr.sparse_coo_tensor(select_outer_pos_indices.T, tr.ones_like(select_outer_pos_indices[:, 0], dtype=bool, device=self.device), [1, self.cos_simi_range[0], self.cos_simi_range[1], 1]).to_dense()
        adj_label_arr = tr.where(select_outer_pos_arr, embedding_bin_num+1, adj_label_arr)
        emb_arr = self.bin_pos_arr[tr.squeeze(adj_label_arr, 3)]
        median_emb_arr = tr.permute(median_blur(tr.permute(emb_arr.type(tr.float), [0, 3, 1, 2]), kernel_size =(2*expand_size+1, 2*expand_size+1)), [0, 2, 3, 1]).type(tr.int64)
        flat_median_emb_arr = tr.reshape(median_emb_arr, [-1, 1, 3])
        expand_bin_pos_arr = tr.reshape(self.bin_pos_arr, [1, -1, 3])
        median_emb_dist_arr = tr.sqrt(tr.sum((flat_median_emb_arr - expand_bin_pos_arr).type(tr.float32) ** 2, -1))
        median_emb_label_arr = tr.reshape(tr.argmin(median_emb_dist_arr, -1), [1, self.cos_simi_range[0], self.cos_simi_range[1], 1])
        median_emb_label_arr = tr.where(bg_pos_arr, embedding_bin_num+1, median_emb_label_arr)
        median_emb_label_arr = tr.where(other_pos_arr, embedding_bin_num, median_emb_label_arr)
        median_emb_label_arr = tr.where((median_emb_label_arr>=embedding_bin_num) & sig_pos_arr, embedding_bin_num, median_emb_label_arr)
        adjust_index_arr = self.compute_adjust_index(label_arr, expand_size, bin_dist_cutoff, same_label_ratio)
        new_label_arr = tr.where(adjust_index_arr, median_emb_label_arr, label_arr)
        if tr.cuda.is_available():
            tr.cuda.empty_cache()
        return new_label_arr
    
    def update_label(self, label_arr, neighbor_close_label_fct, signal_cov_score_fct, is_in_ref_score_fct, argmax_prob_score_fct, ave_bin_dist_cutoff, embedding_expand_size, batch_effect_fct):
        score_arr = self.evaluate_score_new(label_arr, neighbor_close_label_fct, signal_cov_score_fct, is_in_ref_score_fct, argmax_prob_score_fct, ave_bin_dist_cutoff, embedding_expand_size, batch_effect_fct)
        label_arr = tr.argmax(score_arr, -1)
        label_arr = tr.unsqueeze(label_arr, -1)
        return label_arr

    def compute_all(self, mid_signal_cutoff=1.0, neighbor_close_label_fct=0.2, signal_cov_score_fct=1.0, is_in_ref_score_fct=1.0, argmax_prob_score_fct=1.0, ave_bin_dist_cutoff=4, embedding_expand_size=3, batch_effect_fct=0.8, update_num=2):
        logging.debug(f"Initiate label ...")
        self.compute_average_signal()
        if tr.sum((self.ave_expr_arr > mid_signal_cutoff/5).type(tr.float32)) < (self.dataset.cell_radius * self.dataset.cell_radius):
            label_arr = (self.dataset.embedding_bin_num + 1) * tr.ones([1, self.cos_simi_range[0], self.cos_simi_range[1], 1])
            self.label_arr = label_arr.type(tr.int64)
            return None

        self.compute_cos_simi()
        self.init_label_arr(low_expr_cutoff=mid_signal_cutoff/3)
        self.init_neighbor_padding_indices(ave_bin_dist_cutoff)
        self.init_close_embedding_bin_score_mask_arr(embedding_expand_size)
        self.compute_signal_cov_score(mid_signal_cutoff)
        self.generate_all_state_by_cos_simi()
        label_arr = self.label_arr

        for index in range(update_num):
            logging.debug(f"Update: {index+1} / {update_num}")
            label_arr = self.update_label(label_arr, neighbor_close_label_fct, signal_cov_score_fct, is_in_ref_score_fct, argmax_prob_score_fct, ave_bin_dist_cutoff, embedding_expand_size, batch_effect_fct)
            label_arr = self.adjust_label_arr(label_arr)

        self.label_arr = label_arr
        self.generate_all_state_by_cos_simi(use_other=True)
        self.score_arr = self.evaluate_score_new(label_arr, neighbor_close_label_fct, signal_cov_score_fct, is_in_ref_score_fct, argmax_prob_score_fct, ave_bin_dist_cutoff, embedding_expand_size, batch_effect_fct)

    def iter_result(self, skip_bg, mid_signal_cutoff):
        if self.score_arr is not None:
            logging.debug("Interpret results")
            other_score_arr = self.score_arr[0, :, :, self.dataset.embedding_bin_num].to("cpu").numpy()
            bg_score_arr = self.score_arr[0, :, :, self.dataset.embedding_bin_num+1].to("cpu").numpy()
            argmax_emb_score_arr, argmax_emb_arr = tr.max(self.score_arr[0, :, :, :self.dataset.embedding_bin_num].to("cpu"), -1)
            argmax_emb_score_arr = argmax_emb_score_arr.numpy()
            argmax_emb_arr = argmax_emb_arr.numpy()

            sp_label_arr = (self.label_arr+1).to_sparse_coo().to("cpu")
            pos_arr = sp_label_arr.indices().T.numpy()
            label_arr = sp_label_arr.values().numpy() - 1
            embedding_info = self.dataset.embedding_info[["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex", "MainCellType"]].to_numpy()
            embedding_info = np.insert(embedding_info, self.dataset.embedding_bin_num, [np.nan, np.nan, np.nan, "Other"], 0)
            embedding_info = np.insert(embedding_info, self.dataset.embedding_bin_num+1, [np.nan, np.nan, np.nan, "Background"], 0)
            label_embedding_info = embedding_info[label_arr, :]
            cell_type_li = embedding_info[:, 3]
            emb1_arr = self.dataset.embedding_info["Embedding1BinIndex"].to_numpy()
            emb2_arr = self.dataset.embedding_info["Embedding2BinIndex"].to_numpy()
            emb3_arr = self.dataset.embedding_info["Embedding3BinIndex"].to_numpy()
            bin_index_li = self.dataset.embedding_info["BinIndex"].to_numpy()
            ave_expr_arr = self.ave_expr_arr.to("cpu")[0, :, :, 0]
            for embedding_state, (_, x_index, y_index, _), (embedding1_bin_index, embedding2_bin_index, embedding3_bin_index, LabelTransfer)  in zip(label_arr, list(pos_arr), list(label_embedding_info)):
                if skip_bg and (LabelTransfer == "Background") and (ave_expr_arr[x_index, y_index]<(mid_signal_cutoff/10)):
                        continue
                tmp_other_score = other_score_arr[x_index, y_index]
                tmp_bg_score = bg_score_arr[x_index, y_index]
                tmp_bin_index = argmax_emb_arr[x_index, y_index]
                tmp_emb1 = emb1_arr[tmp_bin_index]
                tmp_emb2 = emb2_arr[tmp_bin_index]
                tmp_emb3 = emb3_arr[tmp_bin_index]
                tmp_res_bin_index = bin_index_li[tmp_bin_index]
                if LabelTransfer in ["Background", "Other"]:
                    tmp_emb_score_arr = argmax_emb_score_arr[x_index, y_index]
                    tmp_emb_cell_type = cell_type_li[argmax_emb_arr[x_index, y_index]]
                else:
                    tmp_emb_score_arr = self.score_arr[0, x_index, y_index, embedding_state].to("cpu").numpy()
                    tmp_emb_cell_type = LabelTransfer
                if tmp_emb_score_arr < tmp_bg_score:
                    LabelTransfer = "Background"
                elif tmp_emb_score_arr < tmp_other_score:
                    LabelTransfer = "Other"
                res = [
                    x_index, y_index, tmp_res_bin_index, 
                    tmp_emb1*self.dataset.embedding_resolution, tmp_emb2*self.dataset.embedding_resolution, tmp_emb3*self.dataset.embedding_resolution, 
                    LabelTransfer, tmp_emb_cell_type, tmp_emb_score_arr, tmp_other_score, tmp_bg_score
                    ]
                yield res


class RedeVizImgBinModel(RedeVizBinModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_simi_arr = None
        self.delta_simi_arr = None

    def compute_cos_simi_worker(self):
        delta_expand_size = self.max_bin_expand_size
        simi_arr, bg_simi_arr = self.dataset.compute_cos_simi(self.spot_data)
        simi_arr = simi_arr[:, delta_expand_size: (-1*delta_expand_size), delta_expand_size: (-1*delta_expand_size), :, :]
        bg_simi_arr = bg_simi_arr[:, delta_expand_size: (-1*delta_expand_size), delta_expand_size: (-1*delta_expand_size), :, :]
        return simi_arr, bg_simi_arr
    
    def compute_cos_simi(self) -> None:
        simi_arr, bg_simi_arr = self.compute_cos_simi_worker()
        log2SNR_arr = simi_arr - bg_simi_arr
        self.cos_simi_arr = log2SNR_arr
        self.max_cos_simi_arr, self.argmax_cos_simi_arr = tr.max(log2SNR_arr, 3)
        self.log2SNR_arr = log2SNR_arr

    def init_label_arr_worker(self, low_expr_cutoff=0.3, log2SNR_cutoff=0):
        total_log2SNR_arr = tr.sum(self.log2SNR_arr, -1)
        ave_expr_arr = self.ave_expr_arr
        is_low_expr = ave_expr_arr<low_expr_cutoff
        min_cos_simi_cutoff = tr.min(self.dataset.quantile_cos_simi[:, :, 1, :], 2)[0]
        flat_argmax_cos_simi_arr = tr.reshape(self.argmax_cos_simi_arr, [(self.x_range-2*self.max_bin_expand_size) * (self.y_range-2*self.max_bin_expand_size), self.dataset.bin_num])
        sp_flat_argmax_cos_simi_arr = (flat_argmax_cos_simi_arr+1).to_sparse_coo()
        sp_flat_argmax_cos_simi_arr_indices = sp_flat_argmax_cos_simi_arr.indices()
        sp_flat_argmax_cos_simi_arr_values = sp_flat_argmax_cos_simi_arr.values() - 1
        min_cos_simi_indices = tr.stack([sp_flat_argmax_cos_simi_arr_values, sp_flat_argmax_cos_simi_arr_indices[1, :]], -1)
        min_cos_simi_values = min_cos_simi_cutoff[list(min_cos_simi_indices.T)]
        min_cos_simi_arr = tr.reshape(min_cos_simi_values, [1, self.x_range-2*self.max_bin_expand_size, self.y_range-2*self.max_bin_expand_size, self.dataset.bin_num])
        low_cos_simi_arr = tr.all(self.max_cos_simi_arr < min_cos_simi_arr, -1, keepdim=True)
        max_arr, which_max_arr = tr.max(total_log2SNR_arr, -1, keepdim=True)
        which_max_arr[low_cos_simi_arr] = self.dataset.embedding_bin_num
        for zero_bin_index in self.dataset.zero_bin_index:
            which_max_arr[which_max_arr==zero_bin_index] = self.dataset.embedding_bin_num
        which_max_arr[max_arr<=log2SNR_cutoff] = self.dataset.embedding_bin_num
        which_max_arr[is_low_expr] = self.dataset.embedding_bin_num + 1
        return which_max_arr
    
    def compute_is_in_ref_cell_score(self, all_state_mid_label_arr: tr.Tensor,  all_state_mid_cos_simi_arr: tr.Tensor, log2SNR_cutoff=0):
        score = all_state_mid_cos_simi_arr - log2SNR_cutoff
        is_not_bg = (all_state_mid_label_arr != (self.dataset.embedding_bin_num+1)).type(tr.float32)
        score = score * is_not_bg
        is_out_class = (all_state_mid_label_arr == self.dataset.embedding_bin_num).type(tr.float32)
        score = score * (is_out_class - 0.5) * -2
        score_arr = tr.mean(score, -1)
        return score_arr

    def update_label(self, label_arr, neighbor_close_label_fct, signal_cov_score_fct, is_in_ref_score_fct, argmax_prob_score_fct, ave_bin_dist_cutoff, embedding_expand_size, batch_effect_fct):
        score_arr = self.evaluate_score_new(label_arr, neighbor_close_label_fct, signal_cov_score_fct, is_in_ref_score_fct, argmax_prob_score_fct, ave_bin_dist_cutoff, embedding_expand_size, batch_effect_fct)
        label_arr = tr.argmax(score_arr, -1)
        label_arr = tr.unsqueeze(label_arr, -1)
        for zero_bin_index in self.dataset.zero_bin_index:
            label_arr[label_arr==zero_bin_index] = self.dataset.embedding_bin_num
        return label_arr
    
    def iter_result(self, skip_bg, mid_signal_cutoff):
        if self.score_arr is not None:
            logging.debug("Interpret results")
            other_score_arr = self.score_arr[0, :, :, self.dataset.embedding_bin_num].to("cpu").numpy()
            bg_score_arr = self.score_arr[0, :, :, self.dataset.embedding_bin_num+1].to("cpu").numpy()
            argmax_emb_score_arr, argmax_emb_arr = tr.max(self.score_arr[0, :, :, :self.dataset.embedding_bin_num].to("cpu"), -1)
            argmax_emb_score_arr = argmax_emb_score_arr.numpy()
            argmax_emb_arr = argmax_emb_arr.numpy()

            sp_label_arr = (self.label_arr+1).to_sparse_coo().to("cpu")
            pos_arr = sp_label_arr.indices().T.numpy()
            label_arr = sp_label_arr.values().numpy() - 1
            embedding_info = self.dataset.embedding_info[["Embedding1BinIndex", "Embedding2BinIndex", "Embedding3BinIndex", "MainCellType"]].to_numpy()
            embedding_info = np.insert(embedding_info, self.dataset.embedding_bin_num, [np.nan, np.nan, np.nan, "Other"], 0)
            embedding_info = np.insert(embedding_info, self.dataset.embedding_bin_num+1, [np.nan, np.nan, np.nan, "Background"], 0)
            label_embedding_info = embedding_info[label_arr, :]
            cell_type_li = embedding_info[:, 3]
            emb1_arr = self.dataset.embedding_info["Embedding1BinIndex"].to_numpy()
            emb2_arr = self.dataset.embedding_info["Embedding2BinIndex"].to_numpy()
            emb3_arr = self.dataset.embedding_info["Embedding3BinIndex"].to_numpy()
            bin_index_li = self.dataset.embedding_info["BinIndex"].to_numpy()
            ave_expr_arr = self.ave_expr_arr.to("cpu")[0, :, :, 0]
            for embedding_state, (_, x_index, y_index, _), (embedding1_bin_index, embedding2_bin_index, embedding3_bin_index, LabelTransfer)  in zip(label_arr, list(pos_arr), list(label_embedding_info)):
                if skip_bg and (LabelTransfer == "Background") and (ave_expr_arr[x_index, y_index]<(mid_signal_cutoff/10)):
                        continue
                tmp_other_score = other_score_arr[x_index, y_index]
                tmp_bg_score = bg_score_arr[x_index, y_index]
                tmp_bin_index = argmax_emb_arr[x_index, y_index]
                tmp_emb1 = emb1_arr[tmp_bin_index]
                tmp_emb2 = emb2_arr[tmp_bin_index]
                tmp_emb3 = emb3_arr[tmp_bin_index]
                tmp_res_bin_index = bin_index_li[tmp_bin_index]
                if LabelTransfer in ["Background", "Other"]:
                    tmp_emb_score_arr = argmax_emb_score_arr[x_index, y_index]
                    tmp_emb_cell_type = cell_type_li[argmax_emb_arr[x_index, y_index]]
                else:
                    tmp_emb_score_arr = self.score_arr[0, x_index, y_index, embedding_state].to("cpu").numpy()
                    tmp_emb_cell_type = LabelTransfer

                if tmp_emb_score_arr < tmp_bg_score:
                    LabelTransfer = "Background"
                elif tmp_emb_score_arr < tmp_other_score:
                    LabelTransfer = "Other"
                if tmp_bin_index in self.dataset.zero_bin_index:
                    tmp_emb1 = np.nan
                    tmp_emb2 = np.nan
                    tmp_emb3 = np.nan
                    tmp_bin_index = np.nan
                    if LabelTransfer != "Background":
                        LabelTransfer = "Other"
                    tmp_emb_cell_type = "Other"
                res = [
                    x_index, y_index, tmp_res_bin_index, 
                    tmp_emb1*self.dataset.embedding_resolution, tmp_emb2*self.dataset.embedding_resolution, tmp_emb3*self.dataset.embedding_resolution, 
                    LabelTransfer, tmp_emb_cell_type, tmp_emb_score_arr, tmp_other_score, tmp_bg_score
                    ]
                yield res
