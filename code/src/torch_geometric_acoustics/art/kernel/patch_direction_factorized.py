"""
(Pre-computation) of visibility matrix and material matrices for patch-direction-factorized (PDF) ART.
"""

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric_acoustics.art.kernel.sparse import (
    compose_radiance_kernel,
    mask_sparse_square_matrix,
    sparse_add,
)
from torch_geometric_acoustics.core import (
    compute_local_orthonomal_matrix,
    discretize_direction,
    find_first_intersection,
    sample_direction,
    sample_from_patch,
)


def reflection_kernel_patch_direction_factorized(
    patch_vertex_coords,
    normal,
    brdfs=["diffuse", "specular"],
    N_patch_points=10,
    N_directions=2**10,
    N_azi=16,
    N_ele=16,
    use_local_direction=True,
    bidirectional=True,
):
    kernel_basis = {}
    global_kernel, average_distance = compute_global_kernel(
        patch_vertex_coords,
        normal,
        N_azi=N_azi,
        N_ele=N_ele,
        N_patch_points=N_patch_points,
        N_directions=N_directions,
        use_local_direction=use_local_direction,
        bidirectional=bidirectional,
    )
    kernel_basis = {"global": global_kernel}
    if brdfs is not None and len(brdfs) > 0:
        local_kernels = compute_local_kernel(
            brdfs=brdfs,
            N_azi=N_azi,
            N_ele=N_ele,
            bidirectional=bidirectional,
        )
        kernel_basis.update(local_kernels)
    return kernel_basis, average_distance


def compute_global_kernel(
    patch_vertex_coords,
    normal,
    N_azi=16,
    N_ele=16,
    N_patch_points=10,
    N_directions=2**10,
    use_local_direction=True,
    bidirectional=True,
):
    device = patch_vertex_coords.device
    num_patches = len(patch_vertex_coords)

    num_discrete_directions = N_ele * N_azi
    num_radiances = num_patches * num_discrete_directions

    R = compute_local_orthonomal_matrix(patch_vertex_coords)

    # ------------- initialize kernel basis -------------
    kernel = SparseTensor(
        row=torch.empty(0, dtype=torch.long, device=device),
        col=torch.empty(0, dtype=torch.long, device=device),
        value=torch.empty(0, device=device),
        sparse_sizes=(num_radiances, num_radiances),
    )

    distances = torch.zeros(num_radiances, device=device)
    hit_counts = torch.zeros(num_radiances, device=device)

    # ------------- sample patch points -------------
    full_patch_points = sample_from_patch(patch_vertex_coords, N=N_patch_points)
    N_patch_points = full_patch_points.shape[1]

    for i in tqdm(range(N_patch_points), desc="Kernel Computation"):
        # ------------- per each patch point -------------
        patch_points = full_patch_points[:, i : i + 1, :]
        patch_points = torch.repeat_interleave(patch_points, N_directions, dim=1)
        patch_points = patch_points.view(-1, 3)

        # ------------- send rays -------------
        direction = sample_direction(N=N_directions, method="grid", device=device)
        direction = direction.view(1, -1, 3).repeat(num_patches, 1, 1)
        direction = direction.view(-1, 3)

        # ------------- intersection test -------------
        any_intersection, source_patch_id, _, distance = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=patch_points,
            direction=direction,
        )

        # ------------- get radiance id -------------
        incident_patch_id = torch.arange(num_patches, device=device)
        incident_patch_id = torch.repeat_interleave(incident_patch_id, N_directions)
        incident_orthonomal_matrix = R.repeat_interleave(N_directions, 0)
        incident_direction_id = discretize_direction(
            direction,
            N_ele=N_ele,
            N_azi=N_azi,
            bidirectional=bidirectional,
            local_orthonomal_matrix=incident_orthonomal_matrix,
        )
        incident_radiance_id = (
            incident_patch_id * num_discrete_directions + incident_direction_id
        )

        source_orthonomal_matrix = R[source_patch_id]
        source_direction_id = discretize_direction(
            -direction,
            N_ele=N_ele,
            N_azi=N_azi,
            bidirectional=bidirectional,
            local_orthonomal_matrix=source_orthonomal_matrix,
        )
        source_radiance_id = (
            source_patch_id * num_discrete_directions + source_direction_id
        )

        _kernel = compose_radiance_kernel(
            row=incident_radiance_id,
            col=source_radiance_id,
            val=any_intersection.float(),
            num_radiances=num_radiances,
        )

        kernel = sparse_add(kernel, _kernel)

        distance = distance[any_intersection]
        incident_radiance_id = incident_radiance_id[any_intersection]
        distances = distances + scatter(
            distance,
            incident_radiance_id,
            dim_size=num_radiances,
            reduce="sum",
        )
        hit_counts = hit_counts + scatter(
            torch.ones_like(distance),
            incident_radiance_id,
            dim_size=num_radiances,
            reduce="sum",
        )

    kernel.storage._value /= N_patch_points

    hit_counts_div = hit_counts.clone()
    hit_counts_div[hit_counts == 0] = 1
    average_distance = distances / hit_counts_div

    return kernel, average_distance


def compute_local_kernel(
    brdfs=["diffuse", "specular"],
    N_azi=16,
    N_ele=16,
    bidirectional=False,
    N_directions=2**10,
    device="cuda",
    num_patches=None,
):
    kernels = {}
    direction = sample_direction(N=N_directions, method="grid", device=device)
    direction = direction.view(-1, 3)
    normal = torch.tensor([[0, 0, 1]], device=device, dtype=torch.float32)

    col = discretize_direction(
        direction, N_ele=N_ele, N_azi=N_azi, bidirectional=bidirectional
    )

    num_discrete_directions = N_ele * N_azi

    cos = (direction * normal).sum(-1)
    cos_abs = cos.abs()

    for brdf in brdfs:
        match brdf:
            case "diffuse" | "diffuse_transmission":
                transmission = True if brdf == "diffuse_transmission" else False
                _cos_abs = scatter(cos_abs, col, dim_size=num_discrete_directions)
                val = _cos_abs.repeat_interleave(num_discrete_directions // 2)
                val = val * (4 / N_directions)
                pos_arange = torch.arange(num_discrete_directions // 2, device=device)
                pos_arange = pos_arange.repeat_interleave(num_discrete_directions // 2)
                neg_arange = torch.arange(
                    num_discrete_directions // 2, num_discrete_directions, device=device
                )
                neg_arange = neg_arange.repeat_interleave(num_discrete_directions // 2)
                row = (
                    [neg_arange, pos_arange]
                    if transmission
                    else [pos_arange, neg_arange]
                )
                row = torch.cat(row)
                _col = torch.arange(num_discrete_directions, device=device)
                _col = _col.repeat_interleave(num_discrete_directions // 2)
                kernels[brdf] = compose_radiance_kernel(
                    row=row,
                    col=_col,
                    val=val,
                    num_radiances=num_discrete_directions,
                    method="dense",
                )

            case "specular" | "specular_transmission":
                transmission = True if brdf == "specular_transmission" else False
                new_direction = (
                    -direction
                    if transmission
                    else 2 * cos[:, None] * normal - direction
                )
                row = discretize_direction(
                    new_direction,
                    N_ele=N_ele,
                    N_azi=N_azi,
                    bidirectional=bidirectional,
                )
                val = torch.ones_like(cos)
                kernels[brdf] = compose_radiance_kernel(
                    row=row,
                    col=col,
                    val=val,
                    num_radiances=num_discrete_directions,
                    method="dense",
                )
    if num_patches is not None:
        kernels = {k: v.repeat(num_patches, 1, 1) for k, v in kernels.items()}
    return kernels


def compose_block_diag(local_kernel):
    device = local_kernel.device
    num_patch, num_direction, _ = local_kernel.shape
    direction_arange = torch.arange(num_direction, device=device)
    row, col = torch.meshgrid(direction_arange, direction_arange, indexing="ij")
    row, col = row.flatten(), col.flatten()
    offsets = torch.arange(num_patch, device=device) * num_direction
    row = (offsets[:, None] + row[None, :]).view(-1)
    col = (offsets[:, None] + col[None, :]).view(-1)
    val = local_kernel.view(-1)
    num_radiances = num_patch * num_direction
    return SparseTensor(
        row=row, col=col, value=val, sparse_sizes=(num_radiances, num_radiances)
    )


def compose_full_kernel(global_kernel, local_kernel, nonzero_radiance_mask):
    r"""
    global_kernel: (num_radiance, num_radiance)
    local_kernel: (num_patch, num_direction, num_direction)
    nonzero_radiance_ids: (num_radiances, 2)
    """
    local_kernel = compose_block_diag(local_kernel)
    local_kernel = mask_sparse_square_matrix(local_kernel, nonzero_radiance_mask)
    full_kernel = local_kernel.matmul(global_kernel)
    return full_kernel


if __name__ == "__main__":
    from tqdm import tqdm

    torch.set_default_device("cuda")
    num_patch, num_direction = 200, 100
    local_kernel = torch.rand(num_patch, num_direction, num_direction)
    for _ in tqdm(range(1000)):
        local_blk = compose_block_diag(local_kernel)
