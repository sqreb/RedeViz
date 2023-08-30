import numpy as np
import pandas as pd
import os
from shapely import wkt
from shapely.geometry import Point
from shapely.prepared import prep
import logging


def point2spot_main(args):
    f_in = args.input
    f_cell_shape_info = args.cell_shape_info
    x_range = args.x_range
    y_range = args.y_range
    f_out = args.output
    gap_size = args.gap_size
    spot_size = args.spot_size
    UMI_per_spot = args.UMI_per_spot

    RNA_point_df = pd.read_csv(f_in, sep="\t")
    x_diff = RNA_point_df["x"] % gap_size - (gap_size/2)
    y_diff = RNA_point_df["y"] % gap_size - (gap_size/2)
    cond = np.sqrt(x_diff ** 2 + y_diff ** 2) <= spot_size
    RNA_point_df["is_in_spot"] = cond
    spot_RNA_point_df = RNA_point_df[RNA_point_df["is_in_spot"]]
    spot_RNA_point_df["spot_x_index"] = np.floor(spot_RNA_point_df["x"] / gap_size)
    spot_RNA_point_df["spot_y_index"] = np.floor(spot_RNA_point_df["y"] / gap_size)

    N_points = len(spot_RNA_point_df)
    mu_total_UMI = UMI_per_spot * (x_range * y_range) / (gap_size * gap_size)
    if N_points > mu_total_UMI:
        select_index = np.random.choice(N_points, int(mu_total_UMI))
        select_spot_df = spot_RNA_point_df.iloc[select_index]
    else:
        select_spot_df = spot_RNA_point_df

    spot_UMI_df = select_spot_df.groupby(["spot_x_index", "spot_y_index", "Gid"]).size().reset_index(name='UMI')
    spot_UMI_df[["spot_x_index", "spot_y_index"]] = spot_UMI_df[["spot_x_index", "spot_y_index"]].astype("int")
    spot_UMI_df.to_csv(os.path.join(f_out, "spot.tsv"), sep="\t", index_label=False, index=False)
    
    cell_shape_df = pd.read_csv(f_cell_shape_info, sep="\t")
    cell_shape_df["index"] = range(len(cell_shape_df))
    cell_shape_li = [wkt.loads(cell_shape) for cell_shape in cell_shape_df["CellShape"].to_numpy()]
    prep_cell_shape_li = list(map(prep, cell_shape_li))
    x_li = [gap_size/2+gap_size*index for index in range(int(np.floor(x_range/gap_size)))]
    y_li = [gap_size/2+gap_size*index for index in range(int(np.floor(y_range/gap_size)))]
    x_index_li = list()
    y_index_li = list()
    cell_name_li = list()
    cell_type_li = list()
    bin_index_li = list()
    has_BinIndex = "BinIndex" in cell_shape_df.columns
    for x_index, x in enumerate(x_li):
        if x_index % 10 == 0:
            logging.info(f"{x_index} / {len(x_li)}")
        cell_shape_df_in_line = cell_shape_df.iloc[np.abs(cell_shape_df["x"].to_numpy()-x)<30000,]
        for y_index, y in enumerate(y_li):
            tmp_shape_df_in_line = cell_shape_df_in_line.iloc[np.abs(cell_shape_df_in_line["y"].to_numpy()-y)<30000,]
            cell_num = len(tmp_shape_df_in_line)
            cell_type = "NA"
            cell_name = "NA"
            bin_index = "NA"
            if cell_num:
                for cell_index in range(cell_num):
                    tmp_cell_index = list(tmp_shape_df_in_line["index"])[cell_index]
                    tmp_cell_shape = prep_cell_shape_li[tmp_cell_index]
                    if tmp_cell_shape.contains(Point(x, y)):
                        cell_name = list(tmp_shape_df_in_line["CellName"])[cell_index]
                        cell_type = list(tmp_shape_df_in_line["CellType"])[cell_index]
                        if has_BinIndex:
                            bin_index = list(tmp_shape_df_in_line["BinIndex"])[cell_index]
                        break
            x_index_li.append(x_index)
            y_index_li.append(y_index)
            cell_name_li.append(cell_name)
            cell_type_li.append(cell_type)
            bin_index_li.append(bin_index)
    if has_BinIndex:
        cell_class_df = pd.DataFrame({
            "spot_x_index": x_index_li,
            "spot_y_index": y_index_li,
            "CellName": cell_name_li,
            "CellType": cell_type_li,
            "BinIndex": bin_index_li
        })
    else:
        cell_class_df = pd.DataFrame({
            "spot_x_index": x_index_li,
            "spot_y_index": y_index_li,
            "CellName": cell_name_li,
            "CellType": cell_type_li
        })
    cell_class_df.to_csv(os.path.join(f_out, "Spot.CellType.tsv"), sep="\t", index_label=False, index=False)
