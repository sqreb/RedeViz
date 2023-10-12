import torch as tr
from redeviz.enhance.io import RedeVizBinIndex, load_spot_data
from redeviz.enhance.bin_model import RedeVizBinModel
from redeviz.enhance.utils import SparseZeroPadding2D, SparseTenserSlice2D, select_device
import logging
import numpy as np

def div_spot(spot_data, total_UMI_arr, expand_size, step=256):
    _, x_range, y_range, gene_num = spot_data.shape
    _, x_min, y_min, _ = tr.min(spot_data.indices(), 1)[0].numpy()
    expand_spot_data = SparseZeroPadding2D(spot_data,expand_size, expand_size)
    expand_spot_data = expand_spot_data.coalesce()
    expand_UMI_arr = tr.permute(tr.nn.ZeroPad2d(expand_size)(tr.permute(total_UMI_arr, [0, 3, 1, 2])), [0, 2, 3, 1])
    for x_start in range(0, x_range, step):
        x_end = min(x_start+step, x_range)
        if x_end < x_min:
            continue
        tmp_x_slice_spot_data = tr.index_select(expand_spot_data, 1, tr.LongTensor(tr.tensor(np.arange(x_start, x_end+2*expand_size), dtype=tr.int64, device=spot_data.device)))
        for y_start in range(0, y_range, step):
            y_end = min(y_start+step, y_range)
            if y_end < y_min:
                continue
            tmp_spot_data = tr.index_select(tmp_x_slice_spot_data, 2, tr.LongTensor(tr.tensor(np.arange(y_start, y_end+2*expand_size), dtype=tr.int64, device=spot_data.device)))
            tmp_spot_data = tmp_spot_data.coalesce()
            if int(tr.sparse.sum(tmp_spot_data)) == 0:
                continue
            tmp_total_UMI_arr = expand_UMI_arr[:, x_start:(x_end+2*expand_size), y_start:(y_end+2*expand_size), :]
            yield (tmp_spot_data, tmp_total_UMI_arr, x_start, y_start)

def get_ave_smooth_cov(total_UMI_arr: tr.Tensor, expand_size=10, denoise_cutoff=0.15):
    total_umi = total_UMI_arr.to_sparse_coo()
    new_indices = total_umi.indices() / expand_size
    new_indices = new_indices.type(tr.int64)
    sm_total_umi = tr.sparse_coo_tensor(new_indices, total_umi.values(), total_UMI_arr.shape)
    sm_total_umi = sm_total_umi.coalesce()
    nzo_total_umi = sm_total_umi.values() / (expand_size * expand_size)
    res = float(tr.mean(nzo_total_umi))
    denoise_nzo_total_umi = nzo_total_umi[nzo_total_umi>(res*denoise_cutoff)]
    res = float(tr.mean(denoise_nzo_total_umi))
    return res

def enhance_main(args):
    with tr.no_grad():
        device = args.device_name
        if device is None:
            device = select_device()

        logging.info("Loding index and spots data ...")
        dataset = RedeVizBinIndex(args.index, args.cell_radius, device)
        spot_data, total_UMI_arr, x_range, y_range = load_spot_data(
            args.spot, args.x_index_label, args.y_index_label, args.gene_name_label, args.UMI_label, args.max_expr_ratio, dataset
        )

        if args.mid_signal_cutoff is None:
            logging.info("Computing global coverage threshold ...")
            args.mid_signal_cutoff = get_ave_smooth_cov(total_UMI_arr)
            logging.info(f"The global coverage threshold is {args.mid_signal_cutoff}")

        is_empty = False
        if args.slice_x is not None:
            assert args.slice_y
            _, x_min, y_min, _ = tr.min(spot_data.indices(), 1)[0].numpy()
            if (x_range <= args.slice_x[0]) | (y_range <= args.slice_y[0]) | (x_min >= args.slice_x[1]) | (y_min >= args.slice_y[1]):
                is_empty = True
            else:
                args.slice_x[1] = min(args.slice_x[1], x_range)
                args.slice_y[1] = min(args.slice_y[1], y_range)
                if min((args.slice_x[1] - args.slice_x[0]), (args.slice_y[1] - args.slice_y[0])) < (4 * dataset.max_bin_size):
                    raise ValueError("Slice region is too small")
                spot_data = SparseTenserSlice2D(spot_data, args.slice_x[0], args.slice_x[1], args.slice_y[0], args.slice_y[1])
                total_UMI_arr =  total_UMI_arr[:, args.slice_x[0]: args.slice_x[1], args.slice_y[0]: args.slice_y[1], :]
                x0 = args.slice_x[0]
                y0 = args.slice_y[0]
        else:
            x0 = 0
            y0 = 0

        if tr.sparse.sum(spot_data) == 0:
            is_empty = True
        
        if is_empty:
            with open(args.output, "w") as f:
                header = ["x", "y", "EmbeddingState", "Embedding1", "Embedding2", "Embedding3", "LabelTransfer", "ArgMaxCellType", "RefCellTypeScore", "OtherCellTypeScore", "BackgroundScore"]
                f.write("\t".join(header)+"\n")
            return None

        if args.ave_bin_dist_cutoff is None:
            args.ave_bin_dist_cutoff = max(2, int(10 / dataset.embedding_resolution))

        expand_size = int((dataset.max_bin_size - 1) / 2) + dataset.cell_radius
        with open(args.output, "w") as f:
            header = ["x", "y", "EmbeddingState", "Embedding1", "Embedding2", "Embedding3", "LabelTransfer", "ArgMaxCellType", "RefCellTypeScore", "OtherCellTypeScore", "BackgroundScore"]
            f.write("\t".join(header)+"\n")
            for tmp_spot_data, tmp_total_UMI_arr, x_start, y_start in div_spot(spot_data, total_UMI_arr, expand_size, step=args.window_size):
                tmp_spot_data = tmp_spot_data.to(device)
                tmp_total_UMI_arr = tmp_total_UMI_arr.to(device)
                x_start = x_start + x0
                y_start = y_start + y0
                x_end = min(x_start+args.window_size-1, x_range-1)
                y_end = min(y_start+args.window_size-1, y_range-1)
                logging.info(f"Spot region: x: {x_start}-{x_end}, y: {y_start}-{y_end}")
                model = RedeVizBinModel(dataset, tmp_spot_data, tmp_total_UMI_arr)
                model.compute_all(args.mid_signal_cutoff, args.neighbor_close_label_fct, args.signal_cov_score_fct, args.is_in_ref_score_fct, args.argmax_prob_score_fct, args.ave_bin_dist_cutoff, args.batch_effect_fct, args.update_num)
                for data in model.iter_result(skip_bg=True):
                    if min(data[0], data[1]) < dataset.cell_radius:
                        continue
                    if min((model.cos_simi_range[0] - data[0]), (model.cos_simi_range[1] - data[1])) <= dataset.cell_radius:
                        continue
                    data[0] = data[0] + x_start - dataset.cell_radius
                    data[1] = data[1] + y_start - dataset.cell_radius
                    f.write("\t".join(list(map(str, data)))+"\n")
                model.detach()
                tmp_spot_data.detach()
                tmp_total_UMI_arr.detach()
                if tr.cuda.is_available():
                    tr.cuda.empty_cache()

