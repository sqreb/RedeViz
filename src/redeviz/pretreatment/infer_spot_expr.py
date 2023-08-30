import numpy as np
import pandas as pd
from redeviz.enhance.enhance import get_ave_smooth_cov
import torch as tr

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

def infer_spot_expr_main(args):
    total_UMI_arr = load_total_UMI(args.spot, args.x_index_label, args.y_index_label, args.UMI_label)
    mid_signal_cutoff = get_ave_smooth_cov(tr.tensor(total_UMI_arr))
    with open(args.output, "w") as f:
        f.write(f"{mid_signal_cutoff}")
