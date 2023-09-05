import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
from scipy.sparse import coo_matrix
from redeviz.posttreatment.utils import filter_pred_df

def plot_phenotype_main(args):
    f_spot = args.input
    f_out = args.output
    keep_other = args.keep_other

    spot_df = pd.read_csv(f_spot, sep="\t")
    spot_df = spot_df[spot_df["RefCellTypeScore"] > spot_df["BackgroundScore"]]
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
    R_arr[other_pos] = 50
    G_arr[other_pos] = 50
    B_arr[other_pos] = 50
    RGB_arr = np.stack([R_arr, G_arr, B_arr], -1)
    RGB_arr = RGB_arr.astype(np.uint8)
    plt.imsave(f_out, cv.rotate(RGB_arr, cv.ROTATE_90_COUNTERCLOCKWISE))
