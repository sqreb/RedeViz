import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from cellpose import models
from scipy.sparse import coo_matrix, save_npz
import cv2 as cv
import os
import skimage
import logging


# def mask_by_cellpose(sm_total_UMI_arr, cell_diameter=30, model_type="cyto"):
#     norm_cyto_arr = 255 * sm_total_UMI_arr / sm_total_UMI_arr.max()
#     norm_cyto_arr = norm_cyto_arr.astype(np.uint8)
#     imgs = [norm_cyto_arr]
#     model = models.Cellpose(gpu=True, model_type=model_type)
#     masks, flows, styles, diams = model.eval(imgs, channels=[0, 0], diameter=cell_diameter)
#     mask_arr = masks[0]
#     flow_arr = flows[0][0]
#     mask_num = np.bincount(mask_arr.reshape([-1]))
#     mask_index_arr = np.arange(len(mask_num))
#     rm_mask_index = np.where(mask_num < (cell_diameter * cell_diameter / 4))
#     mask_index_arr[rm_mask_index] = 0
#     new_mask_arr = mask_index_arr[mask_arr]
#     new_flow_arr = flow_arr.copy()
#     zero_pos_pos = np.where(new_mask_arr == 0)
#     new_flow_arr[zero_pos_pos[0], zero_pos_pos[1], :] = 0
#     return new_mask_arr, new_flow_arr

def mask_by_bin_worker(sm_total_UMI_arr, cell_mask_region, cell_diameter=30):
    n_seg = int(cell_mask_region.sum() / (cell_diameter**2 / 4 * np.pi))
    img_x_range, img_y_range = sm_total_UMI_arr.shape
    if n_seg == 0:
        new_mask_arr = np.zeros((img_x_range, img_y_range), dtype=np.int64)
        new_flow_arr = np.zeros((img_x_range, img_y_range, 3), dtype=np.uint8)
        return new_mask_arr, new_flow_arr

    slic_mask_res = skimage.segmentation.slic(sm_total_UMI_arr, mask=cell_mask_region, n_segments=n_seg, start_label=1, channel_axis=None)

    mask_pos = slic_mask_res > 0
    mask_num = np.bincount(slic_mask_res.reshape([-1]))
    mask_x, mask_y = np.where(mask_pos)
    mask_index = slic_mask_res[mask_pos]

    mask_df = pd.DataFrame({"Index": mask_index, "X": mask_x, "Y": mask_y})
    mid_pos = mask_df.groupby(["Index"])[["X", "Y"]].agg(np.median)

    mid_x = np.array([0] + list(mid_pos["X"]))
    mid_y = np.array([0] + list(mid_pos["Y"]))

    mask_x_diff = mask_x - mid_x[mask_index]
    mask_y_diff = mask_y - mid_y[mask_index]
    rm_px_index = (np.abs(mask_x_diff) > cell_diameter * 4) | (np.abs(mask_y_diff) > cell_diameter * 4)
    mask_index[rm_px_index] = 0
    
    slic_mask_res = coo_matrix((mask_index, (mask_x, mask_y)), sm_total_UMI_arr.shape).toarray()
    mask_pos = slic_mask_res > 0

    if np.sum(mask_pos) == 0:
        new_mask_arr = np.zeros((img_x_range, img_y_range), dtype=np.int64)
        new_flow_arr = np.zeros((img_x_range, img_y_range, 3), dtype=np.uint8)
        return new_mask_arr, new_flow_arr

    mask_num = np.bincount(slic_mask_res.reshape([-1]))
    mask_x, mask_y = np.where(mask_pos)
    mask_index = slic_mask_res[mask_pos]
    total_mask_x = coo_matrix((mask_x, (mask_index, np.zeros_like(mask_index))), (max(mask_index)+1, 1)).toarray()[:, 0]
    total_mask_y = coo_matrix((mask_y, (mask_index, np.zeros_like(mask_index))), (max(mask_index)+1, 1)).toarray()[:, 0]
    ave_mask_x = total_mask_x / mask_num
    ave_mask_y = total_mask_y / mask_num
    mask_x_diff = mask_x - ave_mask_x[mask_index]
    mask_y_diff = mask_y - ave_mask_y[mask_index]
    mask_angles = np.arctan2(mask_y_diff, mask_x_diff)+np.pi
    a = 2
    mask_r = ((np.cos(mask_angles)+1)/a)
    mask_g = ((np.cos(mask_angles+2*np.pi/3)+1)/a)
    mask_b =((np.cos(mask_angles+4*np.pi/3)+1)/a)
    mask_r_arr = coo_matrix((mask_r, (mask_x, mask_y)), sm_total_UMI_arr.shape).toarray()
    mask_g_arr = coo_matrix((mask_g, (mask_x, mask_y)), sm_total_UMI_arr.shape).toarray()
    mask_b_arr = coo_matrix((mask_b, (mask_x, mask_y)), sm_total_UMI_arr.shape).toarray()
    mask_img_arr = np.stack([mask_r_arr, mask_g_arr, mask_b_arr], -1)
    mask_img_arr = (mask_img_arr * 255).astype(np.uint8)

    zero_pos = np.where(slic_mask_res == 0)
    mask_img_arr[zero_pos[0], zero_pos[1], :] = 0

    mask_index_arr = np.arange(len(mask_num))
    rm_mask_index = np.where(mask_num < (cell_diameter * cell_diameter / 4))
    mask_index_arr[rm_mask_index] = 0
    new_mask_arr = mask_index_arr[slic_mask_res]
    new_flow_arr = mask_img_arr.copy()
    zero_pos = np.where(new_mask_arr == 0)
    new_flow_arr[zero_pos[0], zero_pos[1], :] = 0
    return new_mask_arr, new_flow_arr

def mask_by_bin(sm_total_UMI_arr, cell_diameter=30, shift_cutoff=0, window_size=1500):
    nzo_sm_UMI_arr = sm_total_UMI_arr[sm_total_UMI_arr>0]
    nzo_Q50 = np.quantile(nzo_sm_UMI_arr, 0.5)
    sm_total_UMI_arr[sm_total_UMI_arr<(nzo_Q50/5)] = 0
    sm_total_UMI_arr = sm_total_UMI_arr.astype(np.float32)
    norm_sm_total_UMI_arr = 255 * sm_total_UMI_arr / sm_total_UMI_arr.max()
    norm_sm_total_UMI_arr = norm_sm_total_UMI_arr.astype(np.uint8)
    cell_mask_region = cv.adaptiveThreshold(norm_sm_total_UMI_arr, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 50*cell_diameter+1, shift_cutoff)
    cell_mask_region[norm_sm_total_UMI_arr==0] = 0

    mask_arr = np.zeros_like(sm_total_UMI_arr, dtype=np.int64)
    flow_arr = np.zeros([sm_total_UMI_arr.shape[0], sm_total_UMI_arr.shape[1], 3], dtype=np.uint8)

    x_range, y_range = sm_total_UMI_arr.shape
    for x_start in range(0, x_range, window_size):
        x_end = min(x_start + window_size, x_range)
        if((x_end - x_start) < window_size/10):
            continue
        for y_start in range(0, y_range, window_size):
            y_end = min(y_start + window_size, y_range)
            if((y_end - y_start) < window_size/10):
                continue
            logging.info(f"X: {x_start}-{x_end}; Y: {y_start}-{y_end}")
            tmp_sm_total_UMI_arr = sm_total_UMI_arr[x_start:x_end, y_start:y_end]
            tmp_norm_sm_total_UMI_arr = norm_sm_total_UMI_arr[x_start:x_end, y_start:y_end]
            tmp_cell_mask_region = cell_mask_region[x_start:x_end, y_start:y_end]
            expr_spot_num = np.sum(tmp_norm_sm_total_UMI_arr>0)
            if expr_spot_num < (4 * (cell_diameter**2)):
                continue
            tmp_mask_arr, tmp_flow_arr = mask_by_bin_worker(tmp_sm_total_UMI_arr, tmp_cell_mask_region, cell_diameter)
            flow_arr[x_start:x_end, y_start:y_end, :] = tmp_flow_arr
            tmp_mask_arr[tmp_mask_arr>0] = tmp_mask_arr[tmp_mask_arr>0] + mask_arr.max()
            mask_arr[x_start:x_end, y_start:y_end] = tmp_mask_arr
    return mask_arr, flow_arr


def load_total_UMI(fname: str, x_label: str, y_label: str, UMI_label: str):
    df = pd.read_csv(fname, sep="\t")
    x_li = df[x_label].to_numpy()
    y_li = df[y_label].to_numpy()
    UMI_li = df[UMI_label].to_numpy().astype(np.float32)
    x_range = x_li.max() + 1
    y_range = y_li.max() + 1

    total_UMI_arr = np.zeros([x_range, y_range])
    for (x, y, UMI) in zip(x_li, y_li, UMI_li):
        if UMI == 0:
            continue
        total_UMI_arr[x, y] += UMI
    return total_UMI_arr

def mask_ST_data_main(args):
    f_spot = args.spot
    x_index_label = args.x_index_label
    y_index_label = args.y_index_label
    UMI_label = args.UMI_label
    smooth_sd = args.smooth_sd
    cell_diameter = args.cell_diameter
    mask_model = args.mask_model
    shift_cutoff = args.shift_cutoff
    f_out = args.output



    if not os.path.exists(f_out):
        os.makedirs(f_out)

    total_UMI_arr = load_total_UMI(f_spot, x_index_label, y_index_label, UMI_label)
    x_index_li, y_index_li = np.where(total_UMI_arr>0)
    min_x = min(x_index_li)
    min_y = min(y_index_li)

    knrnel_size = 2 * smooth_sd + 1
    total_UMI_arr = total_UMI_arr.astype(np.float32)
    sm_total_UMI_arr = cv.GaussianBlur(total_UMI_arr, (knrnel_size, knrnel_size), smooth_sd)
    nzo_sm_total_UMI_arr = sm_total_UMI_arr[sm_total_UMI_arr>0]
    Q90 = np.quantile(nzo_sm_total_UMI_arr, 0.9)
    sm_total_UMI_arr[sm_total_UMI_arr>Q90] = Q90
    plt.imsave(os.path.join(f_out, "smoothed_UMI.png"), cv.rotate(sm_total_UMI_arr[min_x:, min_y:], cv.ROTATE_90_COUNTERCLOCKWISE))

    if mask_model == "cellpose":
        # mask_arr, flow_arr = mask_by_cellpose(sm_total_UMI_arr, cell_diameter, "cyto")
        raise ValueError('Mask model must be in ["bin"].')
    elif mask_model == "bin":
        mask_arr, flow_arr = mask_by_bin(sm_total_UMI_arr, cell_diameter, shift_cutoff)
    else:
        raise ValueError('Mask model must be in ["bin"].')
    
    plt.imsave(os.path.join(f_out, "mask_pos.png"), cv.rotate(flow_arr[min_x:, min_y:, :], cv.ROTATE_90_COUNTERCLOCKWISE))
    save_npz(file=os.path.join(f_out, "mask"), matrix=coo_matrix(mask_arr))