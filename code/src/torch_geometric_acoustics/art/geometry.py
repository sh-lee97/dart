r"""
Code for computing the integrated geometry term required to convert power to radiance.
"""

import itertools

import numpy as np
import torch
import torch.fft
from torch_scatter import scatter
from torch_sparse import SparseTensor
from tqdm import tqdm

from torch_geometric_acoustics.core import (
    discretize_direction,
    sample_direction,
    sample_from_patch,
)

PI = np.pi


@torch.no_grad()
def integrated_geometry_patch_to_patch(patch_vertex_coords, normal, area, patch_N=16):
    device = patch_vertex_coords.device
    num_patches = len(patch_vertex_coords)

    pairs = list(itertools.combinations(range(num_patches), 2))
    pairs = torch.tensor(pairs, device=device).T
    num_pairs = pairs.shape[1]

    patch_points = sample_from_patch(patch_vertex_coords, patch_N, method="equispaced")
    p_a, p_b = patch_points[pairs[0]], patch_points[pairs[1]]

    area_a, area_b = area[pairs[0]], area[pairs[1]]
    normal_a, normal_b = normal[pairs[0]], normal[pairs[1]]
    num_samples = patch_points.shape[1]

    geometry = torch.zeros(num_pairs, device=device)
    for i in tqdm(range(num_samples), desc="Integrated geometry computation"):
        a_to_b = p_b[:, i : i + 1, :] - p_a
        a_to_b_norm = a_to_b / a_to_b.norm(dim=-1, keepdim=True)

        cos_a = torch.einsum("ijk,ik->ij", a_to_b_norm, normal_a)
        cos_b = torch.einsum("ijk,ik->ij", -a_to_b_norm, normal_b)

        cos_a, cos_b = torch.abs(cos_a), torch.abs(cos_b)
        r_square = a_to_b.square().sum(-1)

        geometry_i = cos_a * cos_b / r_square
        geometry = geometry + geometry_i.mean(-1)

    geometry = geometry * area_a * area_b / num_samples

    row, col = pairs
    row, col = torch.cat([row, col]), torch.cat([col, row])
    val = geometry.repeat(2)
    geometry = SparseTensor(
        row=row, col=col, value=val, sparse_sizes=(num_patches, num_patches)
    )
    geometry = geometry.to_dense().view(-1)
    return geometry


def integrated_geometry_patch_direction(
    area,
    N_azi=16,
    N_ele=16,
    bidirectional=True,
    N_samples=100**2,
):
    # shoot rays,
    device = area.device
    directions = sample_direction(N=N_samples, method="grid", device=device)

    # calc. incident cosine,
    normal = torch.tensor([[0, 0, 1]], device=device)
    cosine = (normal * directions).sum(-1).abs()

    # aggregate incident cosines
    direction_id = discretize_direction(
        directions, N_azi=N_azi, N_ele=N_ele, bidirectional=bidirectional
    )
    cosine = scatter(cosine, direction_id, dim=0, dim_size=N_azi * N_ele)

    # normalize by 4pi, multiply by area
    geometry = cosine[None, :] * area[:, None] / N_samples * (4 * PI)
    geometry = geometry.view(-1)
    return geometry


if __name__ == "__main__":
    area = torch.tensor([1, 2.0])
    integrated_geometry_patch_direction(area, N_azi=16, N_ele=16)


@torch.no_grad()
def compute_distance_ranges(
    patch_vertex_coords,
):
    num_patches = len(patch_vertex_coords)
    distances = (
        patch_vertex_coords[:, None, :, None] - patch_vertex_coords[None, :, None, :]
    )
    distances = distances.square().sum(-1).sqrt()
    distances = distances.view(num_patches * num_patches, -1)
    min_distances = distances.min(dim=-1).values
    max_distances = distances.max(dim=-1).values
    return min_distances, max_distances
