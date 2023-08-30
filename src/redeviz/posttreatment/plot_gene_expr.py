import numpy as np
from scipy.sparse import load_npz
import cv2 as cv
from matplotlib import pyplot as plt

def load_expr_arr(fname):
    if fname is None:
        return None
    expr_mat = load_npz(fname)
    expr_mat = expr_mat.toarray()
    nzo_arr = expr_mat[expr_mat>0]
    cutoff = np.percentile(nzo_arr, 95)
    expr_mat[expr_mat>cutoff] = cutoff
    norm_mat = expr_mat / cutoff * 215
    return norm_mat

def single_panel2multi_panel(x_range, y_range, r_arr=None, g_arr=None, b_arr=None):
    if r_arr is None:
        r_arr = np.zeros((x_range, y_range), dtype=np.float64)
    if g_arr is None:
        g_arr = np.zeros((x_range, y_range), dtype=np.float64)
    if b_arr is None:
        b_arr = np.zeros((x_range, y_range), dtype=np.float64)
    r_arr = r_arr + 40
    g_arr = g_arr + 40
    b_arr = b_arr + 40
    rgb_arr = np.stack([r_arr, g_arr, b_arr], -1)
    rgb_arr = rgb_arr.astype(np.uint8)
    return rgb_arr

def plot_gene_expr_main(args):
    f_R = args.R
    f_G = args.G
    f_B = args.B
    f_out = args.output

    assert not all([x is None for x in [f_R, f_G, f_B]])

    R_expr_arr = load_expr_arr(f_R)
    G_expr_arr = load_expr_arr(f_G)
    B_expr_arr = load_expr_arr(f_B)

    x_range = None
    y_range = None
    for obj in [R_expr_arr, G_expr_arr, B_expr_arr]:
        if obj is not None:
            x_range, y_range = obj.shape
            break
    img_arr = single_panel2multi_panel(x_range, y_range, r_arr=R_expr_arr, g_arr=G_expr_arr, b_arr=B_expr_arr)
    plt.imsave(f_out, cv.rotate(img_arr, cv.ROTATE_90_COUNTERCLOCKWISE))
