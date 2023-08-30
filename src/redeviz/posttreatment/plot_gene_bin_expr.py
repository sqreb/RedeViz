import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix

def mat2image_arr(gid, spot_df, nbg_label_arr, x_range, y_range, gene_name_label, UMI_label, x_label, y_label):
    tmp_spot_df = spot_df[spot_df[gene_name_label]==gid]
    expr_arr = coo_matrix((tmp_spot_df[UMI_label].to_numpy(), (tmp_spot_df[x_label].to_numpy(), tmp_spot_df[y_label].to_numpy())), (x_range+1, y_range+1)).toarray()
    nzo_arr = expr_arr[expr_arr>0]
    cutoff = np.percentile(nzo_arr, 95)
    expr_arr[expr_arr>cutoff] = cutoff
    norm_mat = expr_arr / cutoff
    expr_arr = norm_mat * 215
    expr_arr[~nbg_label_arr] = 0
    return expr_arr

def single_panel2multi_panel(x_range, y_range, nbg_label_arr, r_arr=None, g_arr=None, b_arr=None):
    if r_arr is None:
        r_arr = np.zeros((x_range+1, y_range+1), dtype=np.float64)
    if g_arr is None:
        g_arr = np.zeros((x_range+1, y_range+1), dtype=np.float64)
    if b_arr is None:
        b_arr = np.zeros((x_range+1, y_range+1), dtype=np.float64)
    r_arr = r_arr + 40
    g_arr = g_arr + 40
    b_arr = b_arr + 40
    r_arr[~nbg_label_arr] = 0
    g_arr[~nbg_label_arr] = 0
    b_arr[~nbg_label_arr] = 0
    rgb_arr = np.stack([r_arr, g_arr, b_arr], -1)
    rgb_arr = rgb_arr.astype(np.uint8)
    return rgb_arr

def plot_gene_bin_expr_main(args):
    f_in = args.input
    x_label = args.x_label
    y_label = args.y_label
    gene_name_label = args.gene_name_label
    UMI_label = args.UMI_label
    R = args.R
    G = args.G
    B = args.B
    f_out = args.output

    spot_df = pd.read_csv(f_in, sep="\t")
    spot_df[x_label] = spot_df[x_label] - spot_df[x_label].min()
    spot_df[y_label] = spot_df[y_label] - spot_df[y_label].min()
    x_range = spot_df[x_label].max()
    y_range = spot_df[y_label].max()

    pos_df = spot_df[[x_label, y_label]]
    pos_df = pos_df.drop_duplicates()
    nbg_label_arr = coo_matrix(([True] * pos_df.shape[0], (pos_df[x_label].to_numpy(), pos_df[y_label].to_numpy())), (x_range+1, y_range+1))
    nbg_label_arr = nbg_label_arr.toarray()

    if R is not None:
        R_arr = mat2image_arr(R, spot_df, nbg_label_arr, x_range, y_range, gene_name_label, UMI_label, x_label, y_label)
    if G is not None:
        G_arr = mat2image_arr(G, spot_df, nbg_label_arr, x_range, y_range, gene_name_label, UMI_label, x_label, y_label)
    if B is not None:
        B_arr = mat2image_arr(B, spot_df, nbg_label_arr, x_range, y_range, gene_name_label, UMI_label, x_label, y_label)
    img_arr = single_panel2multi_panel(x_range, y_range, nbg_label_arr, r_arr=R_arr, g_arr=G_arr, b_arr=B_arr)
    plt.imsave(f_out, cv.rotate(img_arr, cv.ROTATE_90_COUNTERCLOCKWISE))

