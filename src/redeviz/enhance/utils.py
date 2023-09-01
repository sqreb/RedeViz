import torch as tr
import numpy as np
import logging


def csr_mat2sparse_tensor(X: np.ndarray):
    coo = X.astype(np.float32).tocoo()
    return tr.sparse_coo_tensor(np.mat([coo.row, coo.col]), coo.data, coo.shape)

def sparse_cnt_mat2pct(cnt_mat):
    shape = cnt_mat.shape
    dim = len(shape)
    cnt_mat = cnt_mat.type(tr.float32)
    cnt_mat = cnt_mat.coalesce()
    total_UMI = tr.sparse.sum(cnt_mat, -1).to_dense() + 1e-9
    cnt_mat_indices = cnt_mat.indices()
    cnt_mat_value = cnt_mat.values()
    if dim == 2:
        fct = tr.gather(total_UMI, 0, cnt_mat_indices[0,:])
    else:
        part1 = tr.flip(tr.cumprod(tr.tensor(shape[:-1][::-1]), dim=0), [0])[1:]
        part2 = tr.ones_like(part1)
        loc_arr = tr.unsqueeze(tr.concat([part1, part2], 0), 0)
        loc_arr = loc_arr.type(tr.int64)
        total_UMI = tr.reshape(total_UMI, [-1])
        total_UMI_index = cnt_mat_indices[0:(dim-1), ].transpose(1, 0)
        flat_loc = tr.sum(loc_arr * total_UMI_index, -1)
        fct = tr.gather(total_UMI, 0, flat_loc)
    norm_cnt_mat =  tr.sparse_coo_tensor(cnt_mat_indices, cnt_mat_value/fct, shape)
    return norm_cnt_mat

def sparse_cnt_log_norm(cnt_mat):
    shape = cnt_mat.shape
    dim = len(shape)
    cnt_mat = cnt_mat.type(tr.float32)
    cnt_mat = cnt_mat.coalesce()
    total_UMI = tr.sparse.sum(cnt_mat, -1).to_dense() + 1e-9
    cnt_mat_indices = cnt_mat.indices()
    cnt_mat_value = cnt_mat.values()
    if dim == 2:
        fct = tr.gather(total_UMI, 0, cnt_mat_indices[0,:])
    else:
        part1 = tr.flip(tr.cumprod(tr.tensor(shape[:-1][::-1]), dim=0), [0])[1:]
        part2 = tr.ones_like(part1)
        loc_arr = tr.unsqueeze(tr.concat([part1, part2], 0), 0)
        loc_arr = loc_arr.type(tr.int64)
        total_UMI = tr.reshape(total_UMI, [-1])
        total_UMI_index = cnt_mat_indices[0:(dim-1), ].transpose(1, 0)
        flat_loc = tr.sum(loc_arr * total_UMI_index, -1)
        fct = tr.gather(total_UMI, 0, flat_loc)
    norm_cnt_mat =  tr.sparse_coo_tensor(cnt_mat_indices, tr.log(1 + 1e3 * cnt_mat_value/fct), shape)
    return norm_cnt_mat

def norm_sparse_cnt_mat(cnt_mat):
    shape = cnt_mat.shape
    dim = len(shape)
    cnt_mat = cnt_mat.type(tr.float32)
    cnt_mat = cnt_mat.coalesce()
    sq_cnt_mat = cnt_mat ** 2
    sq_sum_val = tr.sparse.sum(sq_cnt_mat, -1).to_dense()
    mod = tr.sqrt(sq_sum_val) + 1e-9
    cnt_indices = cnt_mat.indices()
    cnt_value = cnt_mat.values()

    if dim == 2:
        fct = tr.gather(mod, 0, cnt_indices[0, :])
    else:
        mod_index = cnt_indices[0:(dim-1), :]
        fct = mod[list(mod_index)]
    norm_cnt_mat =  tr.sparse_coo_tensor(cnt_indices, cnt_value/fct, shape)
    return norm_cnt_mat

def SparseZeroPadding2D(tensor, x_expand_size, y_expand_size):
    ori_shape = tensor.shape
    indices_arr = tensor.indices()
    value_arr = tensor.values()
    expand_indices_arr = tr.ones_like(indices_arr) * tr.tensor([[0], [x_expand_size], [y_expand_size], [0]], dtype=tr.int64, device=indices_arr.device)
    new_indices_arr = indices_arr + expand_indices_arr
    new_shape = tr.Size([ori_shape[0], ori_shape[1]+2*x_expand_size, ori_shape[2]+2*y_expand_size, ori_shape[3]])
    res = tr.sparse_coo_tensor(new_indices_arr, value_arr, new_shape)
    return res

def SparseTenserSlice2D(tensor, x_start, x_end, y_start, y_end):
    tmp_x_slice = tr.index_select(tensor, 1, tr.LongTensor(tr.tensor(np.arange(x_start, x_end), dtype=tr.int64, device=tensor.device)))
    sliced_tensor = tr.index_select(tmp_x_slice, 2, tr.LongTensor(tr.tensor(np.arange(y_start, y_end), dtype=tr.int64, device=tensor.device)))
    sliced_tensor = sliced_tensor.coalesce()
    return sliced_tensor

def SparseSumPooling2DV2(tensor: tr.Tensor, x_expand_size: int, y_expand_size: int):
    device = tensor.device
    indices_arr = tensor.indices()
    t_indices_arr = indices_arr.T
    value_arr = tensor.values()
    x_bin_size = 2 * x_expand_size + 1
    y_bin_size = 2 * y_expand_size + 1
    expand_indices_arr = tr.unsqueeze(t_indices_arr, 0)
    expand_indices_arr = expand_indices_arr.repeat([x_bin_size*y_bin_size, 1, 1])
    expand_indices_arr = tr.reshape(expand_indices_arr, [x_bin_size, y_bin_size, -1, 4])
    shift_indices_arr = tr.tensor([[x, y, 0, 1] for x in range(x_bin_size) for y in range(y_bin_size)] + [[x, y, 0, 2] for x in range(x_bin_size) for y in range(y_bin_size)], dtype=tr.int64, device=device)
    shift_value_arr = tr.tensor([x for x in range(-1*x_expand_size, x_expand_size+1) for y in range(y_bin_size)] + [y for x in range(x_bin_size) for y in range(-1*y_expand_size, y_expand_size+1)], dtype=tr.int64, device=device)
    shift_arr = tr.sparse_coo_tensor(shift_indices_arr.T, shift_value_arr, [x_bin_size, y_bin_size, 1, 4])
    shift_arr = shift_arr.coalesce()
    shift_arr = shift_arr.to_dense()
    new_indices_arr = tr.reshape(expand_indices_arr+shift_arr, [-1, 4])
    new_value_arr = value_arr.repeat([x_bin_size*y_bin_size])
    cond1 = new_indices_arr[:,1] >= 0
    cond2 = new_indices_arr[:,2] >= 0
    cond3 = new_indices_arr[:,1] < tensor.shape[1]
    cond4 = new_indices_arr[:,2] < tensor.shape[2]
    cond = tr.all(tr.stack([cond1, cond2, cond3, cond4], -1), -1)
    new_indices_arr = new_indices_arr[cond,]
    new_value_arr = new_value_arr[cond]
    res = tr.sparse_coo_tensor(new_indices_arr.T, new_value_arr, tensor.shape)
    res = res.coalesce()
    return res

def flat_coo_sparse_tensor_by_position(tensor: tr.Tensor):
    # N, X, Y, D1, D2, ... -> # N, I, D1, D2, ...
    tensor = tensor.coalesce()
    indices_arr = tensor.indices()
    values_arr = tensor.values()
    flat_index_arr = indices_arr[1,] * tensor.shape[2] + indices_arr[2,]
    new_index_arr = tr.concat([tr.stack([indices_arr[0,], flat_index_arr], 0), indices_arr[3:,]], 0)
    res = tr.sparse_coo_tensor(new_index_arr, values_arr, [tensor.shape[0], tensor.shape[1]*tensor.shape[2], *tensor.shape[3:]])
    res = res.coalesce()
    return res

def recover_flat_coo_sparse_tensor_by_position(tensor: tr.Tensor, x_range: int, y_range: int):
    # N, I, D1, D2, ... - > N, X, Y, D1, D2, ...
    tensor = tensor.coalesce()
    indices_arr = tensor.indices()
    values_arr = tensor.values()
    x_index_arr = indices_arr[1,] // y_range
    y_index_arr = indices_arr[1,] % y_range
    new_index_arr = tr.concat([tr.stack([indices_arr[0,], x_index_arr, y_index_arr], 0), indices_arr[2:,]], 0)
    res = tr.sparse_coo_tensor(new_index_arr, values_arr, [tensor.shape[0], x_range, y_range, *tensor.shape[2:]])
    res = res.coalesce()
    return res

def sparse_onehot(loc_arr: tr.Tensor, num_classes: int):
    loc_arr = loc_arr + 1
    which_max_arr = loc_arr.to_sparse_coo()
    indices_arr = which_max_arr.indices()
    value_arr = which_max_arr.values()
    value_arr = value_arr - 1
    value_arr = tr.unsqueeze(value_arr, -1)
    new_indices_arr = tr.concat([indices_arr.T, value_arr], -1)
    new_value_arr = tr.ones_like(new_indices_arr[:, 0])
    one_hot_arr = tr.sparse_coo_tensor(new_indices_arr.T, new_value_arr, [*loc_arr.shape, num_classes])
    one_hot_arr = one_hot_arr.coalesce()
    return one_hot_arr

def select_device():
    import pynvml
    try:
        pynvml.nvmlInit()
        gpu_num = pynvml.nvmlDeviceGetCount()
        gpu_free_mem_li = list() # GB
        for gpu_id in range(gpu_num):
            handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            free_mem = meminfo.free / 1024 / 1024 / 1024
            gpu_free_mem_li.append(free_mem)
        if gpu_free_mem_li:
            gpu_free_mem_li = np.array(gpu_free_mem_li)
            gpu_index_li = np.arange(gpu_num)
            gpu_mem = gpu_free_mem_li.max()
            max_gpu_index_li = gpu_index_li[gpu_free_mem_li==gpu_mem]
            select_gpu_id = int(np.random.choice(max_gpu_index_li, 1))
            logging.info(f"Select GPU:{select_gpu_id}, free mem: {gpu_mem:.2f} GB")
            return f"cuda:{select_gpu_id}"
    except:
        return "cpu"