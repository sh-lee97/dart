"""
(Pre-computation) of reflection kernel for patch-direction (PD) ART.
"""

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric_acoustics.art.kernel.sparse import (
    compose_radiance_kernel,
    sparse_add,
    sparse_coalesce_2d,
)
from torch_geometric_acoustics.core import (
    compute_local_orthonomal_matrix,
    discretize_direction,
    find_first_intersection,
    reflect_direction,
    sample_direction,
    sample_from_patch,
)


def reflection_kernel_patch_direction(
    patch_vertex_coords,
    normal,
    brdfs=["diffuse", "specular"],
    N_patch_points=10,
    N_directions=2**10,
    N_ele=4,
    N_azi=4,
    use_local_direction=True,
    bidirectional=True,
    dense_output=True,
):
    device = patch_vertex_coords.device
    num_patches = len(patch_vertex_coords)
    num_discrete_directions = N_ele * N_azi
    num_radiances = num_patches * num_discrete_directions

    R = compute_local_orthonomal_matrix(patch_vertex_coords)

    # ------------- initialize kernel basis -------------
    kernel_basis = {}
    for k in brdfs:
        kernel_basis[k] = SparseTensor(
            row=torch.empty(0, dtype=torch.long, device=device),
            col=torch.empty(0, dtype=torch.long, device=device),
            value=torch.empty(0, device=device),
            sparse_sizes=(num_radiances, num_radiances),
        )

    distances = torch.zeros(num_radiances, device=device)
    hit_counts = torch.zeros(num_radiances, device=device)

    # ------------- sample patch points -------------
    full_patch_points = sample_from_patch(
        patch_vertex_coords, N=N_patch_points, method="stratified"
    )
    N_patch_points = full_patch_points.shape[1]

    for i in tqdm(range(N_patch_points), desc="Kernel Computation"):

        # ------------- per each patch point -------------
        patch_points = full_patch_points[:, i : i + 1, :]
        patch_points = torch.repeat_interleave(patch_points, N_directions, dim=1)
        patch_points = patch_points.view(-1, 3)

        # ------------- send rays -------------
        direction = sample_direction(N=N_directions, method="grid", device=device)
        direction = direction.view(1, -1, 3).repeat(num_patches, 1, 1)

        # ------------- intersection test -------------
        any_intersection, source_patch_id, _, distance = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=patch_points,
            direction=direction.view(-1, 3),
        )

        # ------------- get incident radiance id -------------
        incident_orthonomal_matrix = R[source_patch_id]
        source_direction_id = discretize_direction(
            -direction.view(-1, 3),
            N_ele=N_ele,
            N_azi=N_azi,
            bidirectional=bidirectional,
            local_orthonomal_matrix=incident_orthonomal_matrix,
        )
        incident_radiance_id = (
            source_patch_id * num_discrete_directions + source_direction_id
        )

        reflector_patch_id = torch.arange(num_patches, device=device)
        reflector_patch_id = torch.repeat_interleave(reflector_patch_id, N_directions)

        if "specular" in brdfs or "specular_transmission" in brdfs:
            reflector_orthonomal_matrix = R.repeat_interleave(N_directions, 0)

        # ------------- compose kernels -------------
        if "diffuse" in brdfs:
            kernel = _diffuse_kernel(
                direction=direction,
                normal=normal,
                any_intersection=any_intersection,
                incident_radiance_id=incident_radiance_id,
                reflector_patch_id=reflector_patch_id,
                num_radiances=num_radiances,
                N_ele=N_ele,
                N_azi=N_azi,
            )
            kernel_basis["diffuse"] = sparse_add(kernel_basis["diffuse"], kernel)

        if "diffuse_transmission" in brdfs:
            kernel = _diffuse_kernel(
                direction=direction,
                normal=normal,
                any_intersection=any_intersection,
                incident_radiance_id=incident_radiance_id,
                reflector_patch_id=reflector_patch_id,
                num_radiances=num_radiances,
                N_ele=N_ele,
                N_azi=N_azi,
                transmission=True,
            )
            kernel_basis["diffuse_transmission"] = sparse_add(
                kernel_basis["diffuse_transmission"], kernel
            )

        if "specular" in brdfs:
            kernel = _specular_kernel(
                direction=direction,
                normal=normal,
                incident_radiance_id=incident_radiance_id,
                reflector_patch_id=reflector_patch_id,
                reflector_orthonomal_matrix=reflector_orthonomal_matrix,
                any_intersection=any_intersection,
                num_radiances=num_radiances,
                N_ele=N_ele,
                N_azi=N_azi,
                bidirectional=bidirectional,
            )
            kernel_basis["specular"] = sparse_add(kernel_basis["specular"], kernel)

        if "specular_transmission" in brdfs:
            kernel = _specular_kernel(
                direction=direction,
                normal=normal,
                incident_radiance_id=incident_radiance_id,
                reflector_patch_id=reflector_patch_id,
                reflector_orthonomal_matrix=reflector_orthonomal_matrix,
                any_intersection=any_intersection,
                num_radiances=num_radiances,
                N_ele=N_ele,
                N_azi=N_azi,
                bidirectional=bidirectional,
                transmission=True,
            )
            kernel_basis["specular_transmission"] = sparse_add(
                kernel_basis["specular_transmission"], kernel
            )

        # ------------- get average distance -------------
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

    # ------------- average the per-patch point kernel -------------
    for k in kernel_basis:
        kernel_basis[k].storage._value /= N_patch_points
    hit_counts_div = hit_counts.clone()
    hit_counts_div[hit_counts == 0] = 1
    average_distance = distances / hit_counts_div

    # ------------- optional dense output -------------
    if dense_output:
        kernel_basis = {k: v.to_dense() for k, v in kernel_basis.items()}
    return kernel_basis, average_distance


def _signed_diffuse_kernel(
    in_cos,
    incident_radiance_id,
    reflector_patch_id,
    num_radiances,
    N_directions,
    N_ele=16,
    N_azi=16,
    positive=True,
    transmission=False,
):
    num_discrete_directions = N_ele * N_azi
    device = reflector_patch_id.device

    if positive:
        in_cos = in_cos.abs() * (in_cos > 0)
    else:
        in_cos = in_cos.abs() * (in_cos < 0)

    reflector_patch_id, incident_radiance_id, in_cos = sparse_coalesce_2d(
        row=reflector_patch_id,
        col=incident_radiance_id,
        val=in_cos,
    )

    if positive ^ transmission:
        reflected_direction_id = torch.arange(
            num_discrete_directions // 2, device=device
        )
    else:
        reflected_direction_id = torch.arange(
            num_discrete_directions // 2, num_discrete_directions, device=device
        )

    reflected_direction_id = reflected_direction_id.repeat(len(reflector_patch_id))
    reflector_patch_id = reflector_patch_id.repeat_interleave(
        num_discrete_directions // 2
    )
    reflected_radiance_id = (
        reflector_patch_id * num_discrete_directions + reflected_direction_id
    )

    incident_radiance_id = incident_radiance_id.repeat_interleave(
        num_discrete_directions // 2
    )
    val = in_cos.repeat_interleave(num_discrete_directions // 2)
    val = val * (4 / N_directions)

    return compose_radiance_kernel(
        row=reflected_radiance_id,
        col=incident_radiance_id,
        val=val,
        num_radiances=num_radiances,
    )


def _diffuse_kernel(
    direction,
    normal,
    any_intersection,
    incident_radiance_id,
    reflector_patch_id,
    num_radiances,
    N_ele=16,
    N_azi=16,
    transmission=False,
):

    # ------------- incident cosine -------------
    in_cos = (direction * normal[:, None, :]).sum(-1)
    in_cos = in_cos.view(-1) * any_intersection
    N_directions = direction.shape[1]

    positive_kernel = _signed_diffuse_kernel(
        in_cos=in_cos,
        incident_radiance_id=incident_radiance_id,
        reflector_patch_id=reflector_patch_id,
        num_radiances=num_radiances,
        N_directions=N_directions,
        N_ele=N_ele,
        N_azi=N_azi,
        positive=True,
        transmission=transmission,
    )
    negative_kernel = _signed_diffuse_kernel(
        in_cos=in_cos,
        incident_radiance_id=incident_radiance_id,
        reflector_patch_id=reflector_patch_id,
        num_radiances=num_radiances,
        N_directions=N_directions,
        N_ele=N_ele,
        N_azi=N_azi,
        positive=False,
        transmission=transmission,
    )
    kernel = sparse_add(positive_kernel, negative_kernel)
    return kernel


def _specular_kernel(
    direction,
    normal,
    incident_radiance_id,
    reflector_patch_id,
    reflector_orthonomal_matrix,
    num_radiances,
    any_intersection,
    N_ele=16,
    N_azi=16,
    bidirectional=True,
    transmission=False,
):
    num_discrete_directions = N_ele * N_azi

    if transmission:
        reflected_direction = -direction
    else:
        reflected_direction = reflect_direction(normal[:, None, :], -direction)
    reflected_direction = reflected_direction.view(-1, 3)

    reflected_direction_id = discretize_direction(
        reflected_direction,
        N_ele=N_ele,
        N_azi=N_azi,
        bidirectional=bidirectional,
        local_orthonomal_matrix=reflector_orthonomal_matrix,
    )
    reflected_radiance_id = (
        reflector_patch_id * num_discrete_directions + reflected_direction_id
    )

    reflected_radiance_count = scatter(
        any_intersection.float(),
        reflected_radiance_id,
        dim_size=num_radiances,
        reduce="sum",
    )
    reflected_radiance_count[reflected_radiance_count == 0] = -1

    normalized_intersection = (
        any_intersection.float() / reflected_radiance_count[reflected_radiance_id]
    )

    row = reflected_radiance_id
    col = incident_radiance_id
    val = normalized_intersection

    return compose_radiance_kernel(row, col, val, num_radiances)
