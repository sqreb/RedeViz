import numpy as np
import pandas as pd


def spot2bin_main(args):
    spot_df = pd.read_csv(args.spot, sep="\t")
    spot_df["bin_x_index"] = np.floor(spot_df[args.x_index_label] / args.bin_size).astype(int)
    spot_df["bin_y_index"] = np.floor(spot_df[args.y_index_label] / args.bin_size).astype(int)
    bin_UMI_df = spot_df.groupby(["bin_x_index", "bin_y_index", args.gene_name_label]).agg({args.UMI_label: sum})
    bin_UMI_df = bin_UMI_df.reset_index()
    bin_UMI_df.to_csv(args.output, sep="\t", index_label=False, index=False)
