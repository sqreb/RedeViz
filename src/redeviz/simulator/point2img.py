import numpy as np
import pandas as pd
import tifffile
import os
from scipy.integrate import dblquad
from scipy.sparse import lil_matrix
import logging

class Intensity(object):
    intensity_cache = dict()

    @classmethod
    def aperture_diffraction_intensity(cls, delta_x: float, delta_y: float, resolution: float):
        a = np.pi * delta_x / (2 * resolution)
        b = np.pi * delta_y / (2 * resolution)
        if a == 0:
            Ia = 1
        else:
            Ia = ((np.sin(a)/a) ** 2)
        if b == 0:
            Ib = 1
        else:
            Ib = ((np.sin(b)/b) ** 2)
        I = Ia * Ib
        return I

    @classmethod
    def intensity_convolution(cls, delta_x: float, delta_y: float, resolution: float):
        if not (delta_x, delta_y, resolution) in cls.intensity_cache.keys():
            x_start_pos = np.array([-2, -1, 0, 1, 2]) * resolution - delta_x - resolution / 2
            x_end_pos = x_start_pos + resolution
            y_start_pos = np.array([-2, -1, 0, 1, 2]) * resolution - delta_y - resolution / 2
            y_end_pos = y_start_pos + resolution
            intensity_mat = np.zeros([5, 5])
            for x_index in range(5):
                for y_index in range(5):
                    intensity_mat[y_index, x_index] = dblquad(
                        lambda x, y: cls.aperture_diffraction_intensity(x, y, resolution),
                        y_start_pos[y_index], y_end_pos[y_index],
                        lambda x: x_start_pos[x_index], lambda x: x_end_pos[x_index]
                        )[0]
            cls.intensity_cache[(delta_x, delta_y, resolution)] = intensity_mat
        return cls.intensity_cache[(delta_x, delta_y, resolution)]
    

def point2img_main(args):
    resolution = args.resolution
    max_UMI_per_px = args.max_UMI_per_px
    x_range = args.x_range
    y_range = args.y_range
    f_out = args.output

    x_px_num = int(np.ceil(x_range / resolution))
    y_px_num = int(np.ceil(y_range / resolution))

    barcode_df = pd.read_csv(args.barcode, sep="\t")
    barcode_name = barcode_df.columns[1:]
    barcode_mat = barcode_df[barcode_name].to_numpy()
    barcode_len = len(barcode_name)
    barcode_gene = list(barcode_df["Gid"])
    barcode_gene2index = {barcode_gene[index]: np.where(barcode_mat[index,:])[0] for index in range(len(barcode_gene))}

    RNA_point_df = pd.read_csv(args.input, sep="\t")
    RNA_point_df["delta_x"] = RNA_point_df["x"] % resolution - resolution / 2
    RNA_point_df["delta_y"] = RNA_point_df["y"] % resolution - resolution / 2
    RNA_point_df["x_index"] = np.floor(RNA_point_df["x"] / resolution)
    RNA_point_df["y_index"] = np.floor(RNA_point_df["y"] / resolution)
    RNA_point_num = RNA_point_df.shape[0]
    max_intensity = Intensity.intensity_convolution(0, 0, resolution).max()

    img_mat_li = [lil_matrix((x_px_num+4, y_px_num+4), dtype=np.float32) for _ in range(barcode_len)]
    for index in range(RNA_point_num):
        if index % 100 == 0:
            logging.info(index)
        gid = RNA_point_df["Gid"][index]
        if gid not in barcode_gene:
            continue

        x_index = int(RNA_point_df["x_index"][index])
        y_index = int(RNA_point_df["y_index"][index])
        delta_x = int(RNA_point_df["delta_x"][index]*2) / 2
        delta_y = int(RNA_point_df["delta_y"][index]*2) / 2
        rel_intensity_mat = Intensity.intensity_convolution(delta_x, delta_y, resolution) / max_intensity
        for barcode_index in barcode_gene2index[gid]:
            img_mat_li[barcode_index][(x_index):(x_index+5),(y_index):(y_index+5)] += rel_intensity_mat

    for index in range(barcode_len):
        img_mat = img_mat_li[index]
        img_mat = img_mat[2:(x_px_num+2), 2:(y_px_num+2)]
        img_mat[img_mat>max_UMI_per_px] = max_UMI_per_px
        img_mat = img_mat * 255 / max_UMI_per_px
        img_mat = img_mat.astype(np.uint8)
        tifffile.imsave(
            os.path.join(f_out, f"{barcode_name[index]}.tiff"), 
            img_mat.toarray(),
            compression='zlib',
            compressionargs={'level': 8},
            )
