import cv2 as cv
from scipy.sparse import coo_matrix
import numpy as np


def filter_pred_df(nbg_spot_df, min_spot_in_region=10):
    pos_arr = coo_matrix((np.arange(nbg_spot_df.shape[0])+1, (nbg_spot_df["x"].to_numpy(), nbg_spot_df["y"].to_numpy())), [nbg_spot_df["x"].max()+1, nbg_spot_df["y"].max()+1])
    pos_arr = pos_arr.toarray()
    ret, markers = cv.connectedComponents(np.uint8(pos_arr>0), connectivity=4)
    label_cnt = np.bincount(np.reshape(markers, [-1]))
    marker_total_cnt_arr = label_cnt[markers]
    nzo_pos = pos_arr>0
    nzo_df_index = pos_arr[nzo_pos]
    nzo_total_cnt_arr = marker_total_cnt_arr[nzo_pos]
    nzo_order = np.argsort(nzo_df_index)
    s_nzo_total_cnt_arr = nzo_total_cnt_arr[nzo_order]
    select_nbg_spot_df = nbg_spot_df[s_nzo_total_cnt_arr>=min_spot_in_region]
    return select_nbg_spot_df