import numpy as np
import os
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
import torch as tr
import logging
from redeviz.posttreatment.utils import filter_pred_df

def label2domain(data):
    embedding_label, cell_coords, label_li, ori_bg_coords = data
    domain_id_arr = np.zeros_like(embedding_label, dtype=int)
    cv.setNumThreads(20)
    for index, label in enumerate(label_li):
        if index % 5 == 0:
            logging.info(f"{index} / {len(label_li)} ...")
        tmp_coords = (embedding_label == label) & cell_coords
        if not tmp_coords.any():
            continue
        ret, markers = cv.connectedComponents(np.uint8(tmp_coords), connectivity=4)
        markers[ori_bg_coords] = 0
        tmp_domain_pos = markers > 0
        domain_id_arr[tmp_domain_pos] = markers[tmp_domain_pos] + domain_id_arr.max()
    domain_id_arr = domain_id_arr - 1
    return domain_id_arr

class SpatialSpace(object):
    def __init__(self, f_spot, keep_other, denoise, min_spot_num) -> None:
        spot_df = pd.read_csv(f_spot, sep="\t")
        spot_df = spot_df[spot_df["RefCellTypeScore"] > spot_df["BackgroundScore"]]
        spot_df = spot_df[spot_df["ArgMaxCellType"]!="Other"]
        self.keep_other = keep_other
        no_bg_spot = spot_df[spot_df["LabelTransfer"]!="Background"]
        if denoise:
            no_bg_spot = filter_pred_df(no_bg_spot, min_spot_in_region=min_spot_num)
    
        self.shift_x = no_bg_spot["x"].min()
        self.shift_y = no_bg_spot["y"].min()
        spot_df["x"] = spot_df["x"] - no_bg_spot["x"].min()
        spot_df["y"] = spot_df["y"] - no_bg_spot["y"].min()
        spot_df = spot_df[spot_df["x"]>=0]
        spot_df = spot_df[spot_df["y"]>=0]
        self.spot_df = spot_df
        self.x_range = self.spot_df["x"].to_numpy().max() + 1
        self.y_range = self.spot_df["y"].to_numpy().max() + 1
        self.bg_pos_arr = None
        self.other_pos_arr = None
        self.embedding_pos_arr = None
        self.domain_index_arr = None
        self.domain_info_df = None
        self.ori_bg_coords = None

    def indices2pos(self, indices: tr.Tensor, num: int, shape: tr.Size) -> tr.Tensor:
        pos_arr = tr.sparse_coo_tensor(indices.T, tr.ones(num, dtype=tr.int64), shape).coalesce()
        return pos_arr

    def build_embedding_pos_arr(self, embedding_spot_indices: tr.Tensor, embedding_spot_embedding_pos: tr.Tensor, shape: tr.Size) -> tr.Tensor:
        embedding1_arr = tr.sparse_coo_tensor(embedding_spot_indices.T, embedding_spot_embedding_pos[:, 0], shape).coalesce().to_dense()
        embedding2_arr = tr.sparse_coo_tensor(embedding_spot_indices.T, embedding_spot_embedding_pos[:, 1], shape).coalesce().to_dense()
        embedding3_arr = tr.sparse_coo_tensor(embedding_spot_indices.T, embedding_spot_embedding_pos[:, 2], shape).coalesce().to_dense()
        embedding_arr = tr.concat([embedding1_arr, embedding2_arr, embedding3_arr], -1)
        return embedding_arr

    def spot_df2pos(self) -> None:
        logging.info("Loading data...")
        self.spot_df["Zero"] = 0
        pos_shape = tr.Size([1, self.x_range, self.y_range, 1])
        embedding_spot_indices = tr.tensor(self.spot_df[["Zero", "x", "y", "Zero"]].to_numpy())
        embedding_spot_embedding_pos = tr.tensor(self.spot_df[["Embedding1", "Embedding2", "Embedding3"]].to_numpy())
        self.embedding_pos_arr = self.build_embedding_pos_arr(embedding_spot_indices, embedding_spot_embedding_pos, pos_shape)

        nbg_df = self.spot_df[self.spot_df["LabelTransfer"]!="Background"]
        nbg_indices = tr.tensor(nbg_df[["Zero", "x", "y", "Zero"]].to_numpy())
        nbg_pos_arr = self.indices2pos(nbg_indices, nbg_indices.shape[0], pos_shape)
        self.bg_pos_arr = (1-nbg_pos_arr.to_dense()).to_sparse_coo()

        if self.keep_other:
            self.other_pos_arr = tr.zeros(pos_shape, dtype=tr.int64).to_sparse_coo()
        else:
            other_cell_type_indices = tr.tensor(self.spot_df[["Zero", "x", "y", "Zero"]][nbg_df["LabelTransfer"].to_numpy() == "Other"].to_numpy())
            self.other_pos_arr = self.indices2pos(other_cell_type_indices, other_cell_type_indices.shape[0], pos_shape)

        ori_bg_coords = self.bg_pos_arr.to_dense().squeeze(0).squeeze(-1).numpy()
        self.ori_bg_coords = ori_bg_coords > 0

    def pos2meanShift_input(self):
        embedding_arr = tr.where(self.embedding_pos_arr>=0, self.embedding_pos_arr, 0) + 20
        embedding_arr = embedding_arr.numpy()
        bg_pos_arr = (self.bg_pos_arr.to_dense().numpy()>0)[:, :, :, 0]
        embedding_arr[bg_pos_arr] = [0, 0, 0]
        other_pos_arr = (self.other_pos_arr.to_dense().numpy()>0)[:, :, :, 0]
        embedding_arr[other_pos_arr] = [255, 255, 255]
        res = tr.tensor(embedding_arr, dtype=tr.float32)
        return res

    def smooth(self, coord_arr, smooth_radius=2):
        logging.info("Smoothing the image ...")
        knrnel_size = 2 * smooth_radius + 1
        sm_coord_arr = cv.GaussianBlur(coord_arr, (knrnel_size, knrnel_size), 0)
        sm_coord_arr[self.ori_bg_coords] = [0, 0, 0]
        return sm_coord_arr

    def adj_mean_shift_res(self, res):
        total_score = res[:, :, :3].sum(-1)
        cond1 = total_score <= 60
        cond2 = total_score >= ((255-20)*3)
        res[cond1, :] = [0, 0, 0, 255]
        res[cond2, :] = [255, 255, 255, 255]
        res[:, :, 3] = 255
        return res

    def run_mean_shift(self, coord_arr, receptive_radius=10, embedding_radius=10, merge_embedding_dist=10, min_spot_num=100):
        logging.info("Run mean-shift filtering ...")
        exp_coord_arr = np.ones([coord_arr.shape[0], coord_arr.shape[1], 4], dtype=np.uint8) * 255
        exp_coord_arr[:, :, :3] = coord_arr
        mean_shift_filter_res = cv.cuda.meanShiftFiltering(cv.cuda_GpuMat(exp_coord_arr), int(receptive_radius), int(embedding_radius))
        mean_shift_filter_res = mean_shift_filter_res.download()
        mean_shift_filter_res = self.adj_mean_shift_res(mean_shift_filter_res)
        mean_shift_seg = cv.cuda.meanShiftSegmentation(cv.cuda_GpuMat(mean_shift_filter_res), max(5, int(receptive_radius/2)), int(merge_embedding_dist), int(min_spot_num))
        mean_shift_seg = mean_shift_seg.download()
        mean_shift_seg = self.adj_mean_shift_res(mean_shift_seg)
        return mean_shift_seg

    def mean_shift2domain(self, mean_shift_seg, min_spot_num=100, thread=20):
        logging.info("Split mean shift results into regions")
        bg_coords = mean_shift_seg[:, :, 2] == 0
        bg_coords = bg_coords | self.ori_bg_coords
        other_coords = mean_shift_seg[:, :, 2] == 255
        cell_coords = (mean_shift_seg[:, :, 2] > 0) & (mean_shift_seg[:, :, 2] < 255)
        embedding1 = mean_shift_seg[:, :, 0]
        embedding2 = mean_shift_seg[:, :, 1]
        embedding3 = mean_shift_seg[:, :, 2]
        embedding1_str = embedding1.astype(str)
        embedding2_str = embedding2.astype(str)
        embedding3_str = embedding3.astype(str)

        embedding_label = np.char.add(embedding1_str, "-")
        embedding_label = np.char.add(embedding_label, embedding2_str)
        embedding_label = np.char.add(embedding_label, "-")
        embedding_label = np.char.add(embedding_label, embedding3_str)

        uniq_embedding_label = np.unique(embedding_label)

        ori_embedding1_arr = np.array(self.embedding_pos_arr[0, :, :, 0])
        ori_embedding2_arr = np.array(self.embedding_pos_arr[0, :, :, 1])
        ori_embedding3_arr = np.array(self.embedding_pos_arr[0, :, :, 2])

        data = (embedding_label, cell_coords, uniq_embedding_label, self.ori_bg_coords)
        domain_id_arr = label2domain(data)
        domain_id_arr[bg_coords] = -2
        domain_id_arr[other_coords] = -3

        domain_pos_arr = domain_id_arr >= 0
        flat_domain_id_arr = tr.tensor(domain_id_arr[domain_pos_arr])
        flat_emb1_arr = tr.tensor(ori_embedding1_arr[domain_pos_arr])
        flat_emb2_arr = tr.tensor(ori_embedding2_arr[domain_pos_arr])
        flat_emb3_arr = tr.tensor(ori_embedding3_arr[domain_pos_arr])
        domain_num = flat_domain_id_arr.max() + 1
        sp_index = flat_domain_id_arr.reshape([1, -1])
        domain_index = np.arange(domain_num)
        domain_spot_num = tr.sparse_coo_tensor(sp_index, tr.ones_like(flat_domain_id_arr), [domain_num]).coalesce().to_dense()
        ave_emb1 = tr.sparse_coo_tensor(sp_index, flat_emb1_arr, [domain_num]).coalesce().to_dense() / domain_spot_num 
        ave_emb2 = tr.sparse_coo_tensor(sp_index, flat_emb2_arr, [domain_num]).coalesce().to_dense() / domain_spot_num 
        ave_emb3 = tr.sparse_coo_tensor(sp_index, flat_emb3_arr, [domain_num]).coalesce().to_dense() / domain_spot_num 

        domain_info_df = pd.DataFrame({
            "OriDomainID": domain_index,
            "AveEmbedding1": ave_emb1,
            "AveEmbedding2": ave_emb2,
            "AveEmbedding3": ave_emb3,
            "SpotNum": domain_spot_num
        })
        domain_info_df = domain_info_df[domain_info_df["SpotNum"]>=min_spot_num]
        domain_info_df["DomainID"] = np.arange(domain_info_df.shape[0])
        domain_trans_li = np.array([-3, -2, -1] + [-1] * domain_num)
        domain_trans_li[domain_info_df["OriDomainID"].to_numpy()+3] = domain_info_df["DomainID"].to_numpy()
        relabel_domain_id_arr = domain_trans_li[domain_id_arr+3]
        domain_info_df = domain_info_df[["DomainID", "AveEmbedding1", "AveEmbedding2", "AveEmbedding3", "SpotNum"]]
        return relabel_domain_id_arr, domain_info_df

    def write_domain_cell_ratio(self, fname):
        cell_type_df = self.spot_df[(self.spot_df["LabelTransfer"].to_numpy()!="Other") & (self.spot_df["LabelTransfer"].to_numpy()!="Background")]
        cell_type_li = cell_type_df["LabelTransfer"].unique()
        cell_type2index = {cell_type: index for (index, cell_type) in enumerate(cell_type_li)}
        cell_type_arr = coo_matrix(([cell_type2index[cell_type] + 1 for cell_type in  cell_type_df["LabelTransfer"].to_numpy()], (cell_type_df["x"].to_numpy(), cell_type_df["y"].to_numpy())), [self.x_range, self.y_range])
        cell_type_arr = cell_type_arr.toarray()
        cell_type_num = len(cell_type_li)
        header = ["DomainID"] + list(cell_type_li)
        with open(fname, "w") as f:
            f.write("\t".join(header)+"\n")
            for domain_id in self.domain_info_df["DomainID"].to_numpy():
                loc_arr = self.domain_index_arr == domain_id
                domain_cell_type = cell_type_arr[loc_arr] - 1
                domain_cell_type = domain_cell_type[domain_cell_type>=0]
                domain_cell_type_cnt = np.bincount(domain_cell_type, minlength=cell_type_num)
                data = [str(domain_id)] + list(map(str, domain_cell_type_cnt))
                f.write("\t".join(data)+"\n")

    def write_res(self, f_dir):
        logging.info("Writting results ...")
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        self.domain_info_df.to_csv(os.path.join(f_dir, "domain.info.tsv"), sep="\t", index_label=False, index=False)
        domain_index_arr = -2 * np.ones([self.x_range+self.shift_x, self.y_range+self.shift_y], dtype=np.int64)
        domain_index_arr[self.shift_x:, self.shift_y:] = self.domain_index_arr
        np.savetxt(os.path.join(f_dir, "domain.label.tsv"), domain_index_arr, delimiter="\t", fmt="%d")
        np.save(os.path.join(f_dir, "domain.label.npy"), domain_index_arr)

        flat_domain_id = np.reshape(domain_index_arr, [-1]) + 3
        self.domain_info_df["One"] = 1
        embedding_color_arr = self.domain_info_df[["AveEmbedding1", "AveEmbedding2", "AveEmbedding3"]].to_numpy() + 20
        embedding_color_arr = embedding_color_arr.astype(np.uint8)
        embedding_color_arr = np.concatenate([np.array([[255, 255, 255], [0, 0, 0], [255, 255, 255]], dtype=np.uint8), embedding_color_arr], 0)
        flat_color_arr = embedding_color_arr[flat_domain_id]
        color_arr = np.reshape(flat_color_arr, [domain_index_arr.shape[0], domain_index_arr.shape[1], 3])
        img_arr = color_arr
        plt.imsave(os.path.join(f_dir, "domain.Embedding.png"), cv.rotate(img_arr, cv.ROTATE_90_COUNTERCLOCKWISE))
        self.write_domain_cell_ratio(os.path.join(f_dir, "domain2cell_type.spot_num.tsv"))
        # with open(os.path.join(f_dir, "domain.pos.tsv"), "w") as f:
        #     header = "DomainID\tx\ty\n"
        #     f.write(header)
        #     for index in range(domain_index_arr.max()+1):
        #         tmp_arr = domain_index_arr == index
        #         loc_x_arr, loc_y_arr = np.where(tmp_arr)
        #         for x, y in zip(loc_x_arr, loc_y_arr):
        #             f.write(f"{index}\t{x}\t{y}\n")

    def compute_all(self, smooth_radius=2, receptive_radius=10, embedding_radius=10, merge_embedding_dist=5, min_spot_num=100, thread=20):
        self.spot_df2pos()
        coord_arr = self.pos2meanShift_input()
        coord_arr = np.uint8(coord_arr[0, :, :, :].numpy())
        coord_arr = self.smooth(coord_arr, smooth_radius)
        res = self.run_mean_shift(coord_arr, receptive_radius, embedding_radius, merge_embedding_dist, min_spot_num)
        self.domain_index_arr, self.domain_info_df = self.mean_shift2domain(res, min_spot_num*0.9, thread)
        

def segment_main(args):
    f_in = args.input
    f_dir = args.output
    smooth_radius = args.smooth_radius
    receptive_radius = args.receptive_radius
    embedding_radius = args.embedding_radius
    merge_embedding_dist = args.merge_embedding_dist
    min_spot_num = args.min_spot_num
    keep_other = args.keep_other
    thread = 1

    spatial_space = SpatialSpace(f_in, keep_other, args.denoise, args.min_spot_num)
    spatial_space.compute_all(smooth_radius, receptive_radius, embedding_radius, merge_embedding_dist, min_spot_num, thread)
    spatial_space.write_res(f_dir)
