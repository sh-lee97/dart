"""
Some utility functions for sparse operations.
These were introduced due to torch_sparse and torch_scatter not supporting some of the core operations.
These are necessary to perform some precomputations without hitting the memory.
"""

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor


def sparse_coalesce_1d(id, val):
    u_id, u_id_inv = torch.unique(id, return_inverse=True)
    val = scatter(val, u_id_inv, reduce="sum")
    return u_id, val


def sparse_coalesce_2d(row, col, val):
    max_col = col.max() + 1
    id = row * max_col + col
    u_id, val = sparse_coalesce_1d(id, val)
    row, col = u_id // max_col, u_id % max_col
    return row, col, val


def compose_sparse_matrix(row, col, val, sparse_sizes):
    row, col, val = sparse_coalesce_2d(row, col, val)
    sparse_tensor = SparseTensor(row=row, col=col, value=val, sparse_sizes=sparse_sizes)
    return sparse_tensor


def sparse_add(sparse_tensor_a, sparse_tensor_b):
    row_a, col_a, val_a = sparse_tensor_a.coo()
    row_b, col_b, val_b = sparse_tensor_b.coo()
    row = torch.cat([row_a, row_b])
    col = torch.cat([col_a, col_b])
    val = torch.cat([val_a, val_b])
    row, col, val = sparse_coalesce_2d(row, col, val)

    size_row_a, size_col_a = sparse_tensor_a.sparse_sizes()
    size_row_b, size_col_b = sparse_tensor_b.sparse_sizes()
    size_row = max(size_row_a, size_row_b)
    size_col = max(size_col_a, size_col_b)
    sparse_sizes = (size_row, size_col)

    sparse_tensor = SparseTensor(row=row, col=col, value=val, sparse_sizes=sparse_sizes)
    return sparse_tensor


def sparse_sum(sparse_tensors):
    r"""
    Sum a list of sparse tensors
    """
    if len(sparse_tensors) == 0:
        return None
    if len(sparse_tensors) == 1:
        return sparse_tensors[0]
    sparse_tensor = sparse_tensors[0]
    for i in range(1, len(sparse_tensors)):
        sparse_tensor = sparse_add(sparse_tensor, sparse_tensors[i])
    return sparse_tensor


def compose_radiance_kernel(
    row, col, val, num_radiances, method="sparse", normalize=True
):
    match method:
        case "sparse":
            kernel = compose_sparse_matrix(
                row=row, col=col, val=val, sparse_sizes=(num_radiances, num_radiances)
            )
        case "dense":
            kernel_id = row * num_radiances + col
            kernel = scatter(val, kernel_id, dim_size=num_radiances * num_radiances)
            kernel = kernel.view(num_radiances, num_radiances)
    if normalize:
        sum = kernel.sum(1)
        sum[sum == 0] = 1
        kernel = kernel / sum[:, None]
    return kernel


def mask_sparse_square_matrix(sparse_tensor, mask):
    r"""
    Mask a sparse matrix along a given dimension.
    equivalent to matrix[mask, :][:, mask], but for COO format
    """
    row, col, val = sparse_tensor.coo()
    row_match, col_match = mask[row], mask[col]
    match = row_match * col_match
    remaining_row, remaining_col, new_val = row[match], col[match], val[match]
    new_ids = mask.long().cumsum() - 1
    new_row, new_col = new_ids[remaining_row], new_ids[remaining_col]
    new_size = mask.count_nonzero()
    return SparseTensor(
        row=new_row, col=new_col, value=new_val, sparse_sizes=(new_size, new_size)
    )


def postprocess_basis_kernels(kernels, radiance_mask):
    in_sums = [kernel.sum(1) for kernel in kernels.values()]
    in_sums = torch.stack(in_sums, dim=0).sum(0)
    out_sums = [kernel.sum(0) for kernel in kernels.values()]
    out_sums = torch.stack(out_sums, dim=0).sum(0)
    energy_mask = (in_sums > 0) * (out_sums > 0)
    kernel_mask = energy_mask * radiance_mask

    if isinstance(kernels, SparseTensor):
        kernels = {
            k: mask_sparse_square_matrix(v, kernel_mask) for k, v in kernels.items()
        }
    else:
        kernels = {k: v[kernel_mask, :][:, kernel_mask] for k, v in kernels.items()}
    return kernels, kernel_mask


def access_sparse_matrix(sparse_matrix, access_row, access_col):
    r"""
    for a 2d sparse matrix,
    return all values at row[i], col[i]
    """
    row, col, val = sparse_matrix.coo()
    N = sparse_matrix.sparse_sizes()[0]
    id = row * N + col
    sparse_tensor_flattened = torch.sparse_coo_tensor(
        indices=id[None, :], values=val, size=(N * N,)
    )
    access_id = access_row * N + access_col
    vals = torch.index_select(sparse_tensor_flattened, 0, access_id)
    vals = vals.to_dense()
    return vals


def compile_basis_kernels(kernels, brdfs):
    kernels = [kernels[brdf] for brdf in brdfs]
    kernel_sum = sparse_sum(kernels)
    row, col, _ = kernel_sum.coo()
    kernels = [access_sparse_matrix(kernel, row, col) for kernel in kernels]
    kernels = torch.stack(kernels)
    return row, col, kernels
