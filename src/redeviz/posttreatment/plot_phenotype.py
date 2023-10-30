import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix
from redeviz.posttreatment.utils import filter_pred_df
import imageio

def change_hue(RGB_arr, delta_H):
    assert delta_H >=0
    assert delta_H < 360
    RGB_F32_arr = RGB_arr.astype(np.float32) / 255.0
    HSV_arr = cv.cvtColor(RGB_F32_arr, cv.COLOR_BGR2HSV)
    HSV_arr[:,:,0] = (HSV_arr[:,:,0] + delta_H)
    HSV_arr[:,:,0][HSV_arr[:,:,0]>360] = HSV_arr[:,:,0][HSV_arr[:,:,0]>360] - 360
    new_RGB_arr = cv.cvtColor(HSV_arr, cv.COLOR_HSV2BGR) * 255.0
    new_RGB_arr = new_RGB_arr.astype(np.uint8)
    return new_RGB_arr

def RGB2gif(f_gif, RGB_arr):
    img_li = list()
    for delta_H in range(0, 360, 5):
        img_li.append(change_hue(RGB_arr, delta_H))
    imageio.mimsave(f_gif, img_li, "GIF", duration=0.01)
    

def plot_phenotype_main(args):
    f_spot = args.input
    f_out = args.output
    keep_other = args.keep_other
    white_bg = args.white_bg
    GIF = args.GIF

    spot_df = pd.read_csv(f_spot, sep="\t")
    spot_df = spot_df[spot_df["RefCellTypeScore"] > spot_df["BackgroundScore"]]
    spot_df["x"] = spot_df["x"] - spot_df["x"].min()
    spot_df["y"] = spot_df["y"] - spot_df["y"].min()

    x_range = spot_df["x"].max() + 1
    y_range = spot_df["y"].max() + 1
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
    R_arr = coo_matrix((sig_df["Embedding1"].to_numpy().astype(int)+20, (sig_df["x"].to_numpy(), sig_df["y"].to_numpy())), (x_range, y_range)).toarray()
    G_arr = coo_matrix((sig_df["Embedding2"].to_numpy().astype(int)+20, (sig_df["x"].to_numpy(), sig_df["y"].to_numpy())), (x_range, y_range)).toarray()
    B_arr = coo_matrix((sig_df["Embedding3"].to_numpy().astype(int)+20, (sig_df["x"].to_numpy(), sig_df["y"].to_numpy())), (x_range, y_range)).toarray()
    R_arr[other_pos] = 150
    G_arr[other_pos] = 150
    B_arr[other_pos] = 150
    RGB_arr = np.stack([R_arr, G_arr, B_arr], -1)
    RGB_arr = RGB_arr.astype(np.uint8)

    if white_bg:
        bg_pos = RGB_arr.sum(-1) == 0
        RGB_arr[bg_pos] = 255
    if GIF:
        RGB2gif(f_out, RGB_arr)
    else:
        plt.imsave(f_out, cv.rotate(RGB_arr, cv.ROTATE_90_COUNTERCLOCKWISE))
