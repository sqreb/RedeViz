import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix
from redeviz.posttreatment.utils import filter_pred_df

def plot_cell_type_main(args):
    f_spot = args.input
    f_out = args.output
    f_color = args.color
    keep_other = args.keep_other
    white_bg = args.white_bg

    spot_df = pd.read_csv(f_spot, sep="\t")
    spot_df["x"] = spot_df["x"] - spot_df["x"].min()
    spot_df["y"] = spot_df["y"] - spot_df["y"].min()
    x_range = spot_df["x"].max() + 1
    y_range = spot_df["y"].max() + 1
    spot_df = spot_df[spot_df["RefCellTypeScore"] > spot_df["BackgroundScore"]]
    nbg_spot_df = spot_df[spot_df["LabelTransfer"]!="Background"]
    if args.denoise:
        nbg_spot_df = filter_pred_df(nbg_spot_df, min_spot_in_region=args.min_spot_num)
    
    if keep_other:
        sig_df = nbg_spot_df[nbg_spot_df["ArgMaxCellType"]!="Other"]
        other_df = nbg_spot_df[nbg_spot_df["ArgMaxCellType"]=="Other"]
    else:
        sig_df = nbg_spot_df[nbg_spot_df["LabelTransfer"]!="Other"]
        other_df = nbg_spot_df[nbg_spot_df["LabelTransfer"]=="Other"]
        
    other_pos = coo_matrix(([1]*other_df.shape[0], (other_df["x"].to_numpy(), other_df["y"].to_numpy())), (x_range, y_range)).toarray()
    other_pos = other_pos.astype(bool)

    color_df = pd.read_csv(f_color, sep="\t")
    color_df.columns = ["ArgMaxCellType", "R", "G", "B"]
    color_df = color_df.drop_duplicates(["ArgMaxCellType"], keep="first")
    color_cell_type = color_df["ArgMaxCellType"].to_numpy()
    for ct in sig_df["ArgMaxCellType"].unique():
        if not ct in color_cell_type:
            raise ValueError(f"Not found color config for {ct}")
    sig_df = sig_df.merge(color_df)

    other_pos = other_pos.astype(bool)
    R_arr = coo_matrix((sig_df["R"].to_numpy(), (sig_df["x"].to_numpy(), sig_df["y"].to_numpy())), (x_range, y_range)).toarray()
    G_arr = coo_matrix((sig_df["G"].to_numpy(), (sig_df["x"].to_numpy(), sig_df["y"].to_numpy())), (x_range, y_range)).toarray()
    B_arr = coo_matrix((sig_df["B"].to_numpy(), (sig_df["x"].to_numpy(), sig_df["y"].to_numpy())), (x_range, y_range)).toarray()
    R_arr[other_pos] = 150
    G_arr[other_pos] = 150
    B_arr[other_pos] = 150
    RGB_arr = np.stack([R_arr, G_arr, B_arr], -1)
    RGB_arr = RGB_arr.astype(np.uint8)
    if white_bg:
        bg_pos = RGB_arr.sum(-1) == 0
        RGB_arr[bg_pos] = 255
    plt.imsave(f_out, cv.rotate(RGB_arr, cv.ROTATE_90_COUNTERCLOCKWISE))
