"""
(Pre-computation) of reflection kernel for patch-to-patch (P2P) ART.
"""

import torch
from torch_scatter import scatter
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric_acoustics.art.kernel.sparse import (
    compose_radiance_kernel,
    sparse_add,
)
from torch_geometric_acoustics.core import (
    find_first_intersection,
    intersection_test,
    reflect_direction,
    sample_direction,
    sample_from_patch,
)


def reflection_kernel_patch_to_patch(
    patch_vertex_coords,
    normal,
    brdfs=["diffuse", "specular"],
    N_patch_points=4,
    N_directions=2**10,
    dense_output=False,
):
    device = patch_vertex_coords.device
    num_patches = len(patch_vertex_coords)
    num_radiances = num_patches * num_patches

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
        patch_vertex_coords, N=N_patch_points, method="equispaced"
    )
    N_patch_points = full_patch_points.shape[1]

    if "diffuse" in brdfs or "diffuse_transmission" in brdfs:
        # ------------- get patch-to-patch visibility matrix -------------
        mean_visibility = patch_to_patch_mean_visibility(
            patch_vertex_coords, normal, N_patch_points=N_patch_points
        )
        mean_visibility = mean_visibility.view(-1)
        mean_visibility = mean_visibility.repeat(num_patches)

    for i in tqdm(range(N_patch_points), desc="Kernel Computation"):
        # ------------- per each patch point -------------
        patch_points = full_patch_points[:, i : i + 1, :]
        patch_points = torch.repeat_interleave(patch_points, N_directions, dim=1)

        # ------------- send rays -------------
        direction = sample_direction(N=N_directions, method="grid", device=device)
        direction = direction.view(1, -1, 3).repeat(
            num_patches, 1, 1
        )  

        # ------------- intersection test -------------
        any_intersection, source_patch_id, _, distance = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=patch_points.view(-1, 3),
            direction=direction.view(-1, 3),
        )

        # ------------- get incident radiance id -------------
        reflector_patch_id = torch.arange(num_patches, device=device)
        reflector_patch_id = torch.repeat_interleave(reflector_patch_id, N_directions)
        incident_radiance_id = source_patch_id * num_patches + reflector_patch_id

        # ------------- get outgoing radiance id and energy -------------
        if "diffuse" in brdfs:
            kernel = _diffuse_kernel(
                direction,
                normal,
                any_intersection,
                incident_radiance_id,
                num_radiances,
                num_patches,
                mean_visibility,
            )
            kernel_basis["diffuse"] = sparse_add(kernel_basis["diffuse"], kernel)

        if "specular" in brdfs:
            kernel = _specular_kernel(
                patch_points,
                direction,
                normal,
                patch_vertex_coords,
                reflector_patch_id,
                num_radiances,
                num_patches,
                incident_radiance_id,
                any_intersection,
            )
            kernel_basis["specular"] = sparse_add(kernel_basis["specular"], kernel)

        if "diffuse_transmission" in brdfs:
            kernel = _diffuse_kernel(
                direction,
                normal,
                any_intersection,
                incident_radiance_id,
                num_radiances,
                num_patches,
                1 - mean_visibility,
            )
            kernel_basis["diffuse_transmission"] = sparse_add(
                kernel_basis["diffuse_transmission"], kernel
            )

        if "specular_transmission" in brdfs:
            kernel = _specular_kernel(
                patch_points,
                direction,
                normal,
                patch_vertex_coords,
                reflector_patch_id,
                num_radiances,
                num_patches,
                incident_radiance_id,
                any_intersection,
                transmission=True,
            )
            kernel_basis["specular_transmission"] = sparse_add(
                kernel_basis["specular_transmission"], kernel
            )
        #
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


def patch_to_patch_mean_visibility(patch_vertex_coords, normal, N_patch_points=10):
    """
    outputs [num_patches, num_patches] visibility matrix
    where ij entry is portion of patch j visible from positive side of patch i
    """
    num_patches = len(patch_vertex_coords)
    device = patch_vertex_coords.device

    # (num_patches, 3)
    centers = patch_vertex_coords.mean(-2)

    # (num_patches, N_patch_points, 3)
    target_patch_points = sample_from_patch(patch_vertex_coords, N=N_patch_points)
    N_patch_points = target_patch_points.shape[1]

    mean_visibility = torch.zeros(num_patches, num_patches, device=device)
    for i in range(N_patch_points):
        # (num_patches, 3)
        target_patch_points_i = target_patch_points[:, i, :]

        # (num_patches, num_patches, 3)
        center_to_target = target_patch_points_i[None, :, :] - centers[:, None, :]
        prod = (center_to_target * normal[:, None, :]).sum(-1)
        visibility = (prod > 0).float()
        mean_visibility += visibility

    # (num_patches, num_patches)
    mean_visibility = mean_visibility / N_patch_points
    return mean_visibility


def _diffuse_kernel(
    direction,
    normal,
    any_intersection,
    incident_radiance_id,
    num_radiances,
    num_patches,
    mean_visibility,
):
    # ------------- incident cosine -------------
    N_directions = direction.shape[1]

    in_cos = (direction * normal[:, None, :]).sum(-1)
    in_cos = in_cos.view(-1)
    positive_mask = in_cos > 0
    negative_mask = in_cos < 0
    abs_in_cos = torch.abs(in_cos) * any_intersection

    pos_in_cos = abs_in_cos * positive_mask
    pos_in_cos = scatter(pos_in_cos, incident_radiance_id, dim_size=num_radiances)
    pos_in_cos = pos_in_cos.repeat_interleave(num_patches)

    neg_in_cos = abs_in_cos * negative_mask
    neg_in_cos = scatter(neg_in_cos, incident_radiance_id, dim_size=num_radiances)
    neg_in_cos = neg_in_cos.repeat_interleave(num_patches)

    # ------------- compose kernel -------------
    device = direction.device
    radiance_arange = torch.arange(num_radiances, device=device)
    patch_arange = torch.arange(num_patches, device=device)

    reflector_patch_id = (radiance_arange % num_patches).repeat_interleave(num_patches)
    dest_patch_id = patch_arange.repeat(num_radiances)
    row = num_patches * reflector_patch_id + dest_patch_id
    col = radiance_arange.repeat_interleave(num_patches)

    val = pos_in_cos * mean_visibility + neg_in_cos * (1 - mean_visibility)
    val = val * (4 / N_directions)

    return compose_radiance_kernel(
        row=row, col=col, val=val, num_radiances=num_radiances
    )


def _specular_kernel(
    patch_points,
    direction,
    normal,
    patch_vertex_coords,
    reflector_patch_id,
    num_radiances,
    num_patches,
    incident_radiance_id,
    any_intersection,
    transmission=False,
):
    device = patch_vertex_coords.device

    # ------------- reflect direction, intersection test -------------
    if transmission:
        reflected_direction = -direction
    else:
        reflected_direction = reflect_direction(normal[:, None, :], -direction)

    intersection, _ = intersection_test(
        patch_vertex_coords=patch_vertex_coords,
        origin=patch_points.view(-1, 3),
        direction=reflected_direction.view(-1, 3),
    )
    intersection = intersection * any_intersection[:, None]
    intersection = intersection.view(num_patches, -1, num_patches)

    reflected_radiance_count = intersection.sum(-2).view(-1)
    reflected_radiance_count[reflected_radiance_count == 0] = -1

    intersection = intersection.reshape(-1).float()

    # ------------- get kernel id -------------
    arange = torch.arange(num_patches, device=device)
    reflected_radiance_id = reflector_patch_id[:, None] * num_patches + arange[None, :]
    reflected_radiance_id = reflected_radiance_id.view(-1)
    incident_radiance_id = torch.repeat_interleave(incident_radiance_id, num_patches)

    normalized_intersection = (
        intersection / reflected_radiance_count[reflected_radiance_id]
    )

    # ------------- compose kernel -------------
    return compose_radiance_kernel(
        row=reflected_radiance_id,
        col=incident_radiance_id,
        val=normalized_intersection,
        num_radiances=num_radiances,
    )
