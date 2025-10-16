r"""
Core operations for geometric acoustics, written in PyTorch.
"""

import math

import torch
import torch.nn.functional as F

PI = math.pi
PHI = math.pi * (math.sqrt(5) - 1)


def apply_integer_delay(radiance, delay):
    """
    radiance: [N, T], float tensor
    delay: [N], long tensor
    """
    N, T = radiance.shape
    D = delay.max().item()
    assert D > 0
    assert delay.dtype == torch.long
    device = radiance.device

    delayed_radiance = torch.zeros(N, T + D, device=device)
    row_indices = torch.arange(N, device=device).unsqueeze(1)  # Shape (N, 1)
    col_indices = delay.unsqueeze(1) + torch.arange(T, device=device)  # Shape (N, K)
    delayed_radiance[row_indices, col_indices] = radiance
    delayed_radiance = delayed_radiance[:, :-D]
    return delayed_radiance


def delay_impulse(delay_samples, signal_len, method="fraction_sinc"):
    r"""
    delay_samples: [B]
    """
    device = delay_samples.device
    match method:
        case "fraction_sinc":
            arange = torch.arange(signal_len, device=device)
            delay = torch.sinc(arange[None, :] - delay_samples[:, None])

        case "fraction_linear":
            arange = torch.arange(signal_len, device=device)
            delay = F.relu(1 - (arange[None, :] - delay_samples[:, None]).abs())

        case "integer_nearest":
            num_delays = len(delay_samples)
            delay = torch.zeros(num_delays, signal_len, device=device)
            arange = torch.arange(num_delays, device=device)
            delay_samples = delay_samples.round().long()
            delay[arange, delay_samples] = 1.0
    return delay


def reflect_direction(normal, direction):
    r"""
    reflect *incident* direction on a surface with normal.
    """
    len_proj = (direction * normal).sum(-1, keepdim=True)
    return direction - 2 * len_proj * normal


def compute_patch_geometric_properties(
    vertex_coords, patch_vertex_ids, is_triangle=True
):
    patch_vertex_coords = vertex_coords[patch_vertex_ids]  # [M, 3, 3]
    u = patch_vertex_coords[:, 1] - patch_vertex_coords[:, 0]
    v = patch_vertex_coords[:, -1] - patch_vertex_coords[:, 0]
    normal = torch.cross(u, v, -1)
    area = torch.linalg.norm(normal, ord=2, dim=-1)
    if is_triangle:
        area = area / 2
    normal = normal / torch.linalg.norm(normal, ord=2, dim=-1, keepdim=True)
    return patch_vertex_coords, normal, area


# @profile
def find_first_intersection(
    patch_vertex_coords,
    origin,
    direction=None,
    eps=1e-4,
):
    direction = direction / torch.norm(direction, dim=-1, keepdim=True)
    mask, distance = intersection_test(
        patch_vertex_coords=patch_vertex_coords,
        origin=origin,
        direction=direction,
        eps=eps,
    )

    distance[~mask] = float("inf")
    distance[distance <= eps] = float("inf")
    distance, intersection_id = torch.min(distance, dim=-1)

    any_intersection = mask.any(-1)

    intersection_pos = origin + distance[..., None] * direction
    return any_intersection, intersection_id, intersection_pos, distance



def intersection_test(
    patch_vertex_coords,
    origin,
    target=None,
    direction=None,
    is_triangle=True,
    eps=1e-7,
    bidirectional=False,
):
    """
    Moller-Trumbore ray-triangle intersection algorithm.

    Args:
        patch_vertex_coords: torch.Tensor, shape [M, 3, 3], representing the vertices of M triangles
        origin: torch.Tensor, shape [..., 3], representing the start points of (...) = N line segments
        target: torch.Tensor, shape [..., 3], representing the end points of (...) = N line segments
        direction: torch.Tensor, shape [..., 3], representing the end points of (...) = N line segments

    Returns:
        intersection_mask [..., M], boolean tensor indicating intersections
        intersection_point [..., M, 3], intersection points
        distance [..., M], distance from origin to intersection point
    """
    # Number of lines and triangles
    # N = origin.shape[0]
    # M = patch_vertex_coords.shape[0]
    assert (target is None) ^ (direction is None)
    shape = origin.shape[:-1]

    origin = origin.view(-1, 3)
    if target is not None:
        target = target.view(-1, 3)
    if direction is not None:
        direction = direction.view(-1, 3)

    # Unpack triangle vertices
    v1 = patch_vertex_coords[:, 0]  # shape [M, 3]
    v2 = patch_vertex_coords[:, 1]  # shape [M, 3]
    v3 = patch_vertex_coords[:, -1]  # shape [M, 3]

    # Compute edges of the triangles
    edge1 = v2 - v1  # shape [M, 3]
    edge2 = v3 - v1  # shape [M, 3]

    # Compute ray directions for each line segment
    # shape [N, 1, 3]
    if direction is None:
        direction = target - origin
        line_check = True
    else:
        line_check = False
    direction = direction / torch.norm(direction, dim=-1, keepdim=True)
    direction = direction.unsqueeze(1)

    # Cross product of direction with edge2
    # shape [N, M, 3]
    h = torch.cross(direction, edge2.unsqueeze(0), -1)

    # Determinant for each pair (N line segments, M triangles)
    a = torch.einsum("nmj,mj->nm", h, edge1)  # Dot product, shape [N, M]

    # Check for rays parallel to triangles (near-zero determinant)
    parallel = torch.abs(a) < 1e-8  # shape [N, M]

    # Calculate u parameter
    f = 1.0 / a
    s = origin.unsqueeze(1) - v1.unsqueeze(0)  # shape [N, M, 3]
    u = f * torch.einsum("nmj,nmj->nm", s, h)  # shape [N, M]

    # u should be in [0, 1] range
    u_mask = (u < 0) | (u > 1)  # shape [N, M]

    # Calculate v parameter
    q = torch.cross(s, edge1.unsqueeze(0), -1)  # shape [N, M, 3]
    v = f * torch.einsum("nij,nmj->nm", direction, q)  # shape [N, M]

    # v should be in [0, 1] range and u + v should be <= 1
    w = 1 - u - v
    if is_triangle:
        v_mask = (v < 0) | (w < 0)  # <<< FOR TRIANGLE
    else:
        v_mask = (v < 0) | (v > 1)  # <<< FOR PARALLEOGRAM

    # Calculate t parameter to check if intersection happens within line segment
    distance = f * torch.einsum("nmj,mj->nm", q, edge2)  # shape [N, M]

    if not bidirectional:
        # Check if t is within the bounds of the line segment
        distance_mask = distance < eps
        if line_check:
            line_length = torch.norm(target - origin, dim=-1).unsqueeze(1)
            distance_mask = distance_mask | (distance > line_length)

        # Final result: intersection occurs if none of the conditions for no intersection are met
    else:
        # bidirectional; and condition
        distance_mask = (distance < eps) * (distance > -eps)
    intersect_mask = ~(parallel | u_mask | v_mask | distance_mask)

    intersect_mask = intersect_mask.view(*shape, -1)
    distance = distance.view(*shape, -1)
    return intersect_mask, distance



@torch.no_grad()
def sample_quasirandom_r2(N=1000, device="cpu"):
    G = 1.32471795724474602596
    a1 = 1 / G
    a2 = 1 / (G * G)
    arange = 1.5 + torch.arange(N, device=device)
    x = (arange * a1) % 1
    y = (arange * a2) % 1
    samples = torch.stack([x, y], dim=0)
    return samples


@torch.no_grad()
def sample_surface_patch_rays_monte_carlo(
    patch_vertex_coords, N=1000, is_triangle=True
):
    """
    Sample rays from the surface of a patch defined by the vertices.
    Parameters:
        patch_vertex_coords: torch.Tensor, shape [M, 3, 3], coordinates of the M triangle vertices
        N: int, number of rays to sample

    Returns:
        torch.Tensor, shape [M, N, 3], sampled rays
    """
    device = patch_vertex_coords.device
    M = patch_vertex_coords.shape[0]
    direction = sample_direction_from_hemisphere(normal=patch_vertex_coords[:, 0], N=N)
    origin = sample_from_patch(patch_vertex_coords, N=N, is_triangle=is_triangle)
    return origin, direction


@torch.no_grad()
def sample_direction_from_hemisphere(normal, N=1000, method="fibonacci"):
    """
    Uniformly sample directions (unit vectors) from the hemisphere defined by the normal.
    Parameters:
        normal: torch.Tensor, shape [M, 3], normal vectors of the hemisphere

    Returns:
        torch.Tensor, shape [M, N, 3], sampled directions
    """
    device = normal.device
    M = len(normal)

    match method:
        case "fibonacci":
            direction = sample_direction(N=N, device=device, method=method)
            direction = direction[None, :, :].repeat(M, 1, 1)
        case "uniform":
            direction = sample_direction(N=M * N, device=device, method=method)
            direction = direction.view(M, N, 3)
    normal = normal[:, None, :]
    dot = torch.sum(normal * direction, dim=-1)
    direction[dot < 0] = -direction[dot < 0]
    return direction


@torch.no_grad()
def sample_lambertian_rays(normal, N):
    """
    Sample rays from the Lambertian distribution.
    Parameters:
        normal: torch.Tensor, shape [M, 3], normal vectors of the hemisphere
        N: int, number of rays to sample

    Returns:
        torch.Tensor, shape [M, N, 3], sampled rays
    """
    device = normal.device
    M = len(normal)

    r1, r2 = torch.rand(2, M * N, device=device)

    phi = 2 * PI * r1
    theta = torch.acos(torch.sqrt(1 - r2))

    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(theta)

    direction = torch.stack([x, y, z], dim=-1).view(M, N, 3)

    tangent = torch.zeros(M, 3, device=device)
    mask = normal[:, 0].abs() < 0.9
    tangent[mask, 0] = 1.0
    tangent[~mask, 1] = 1.0
    tangent = tangent - torch.sum(tangent * normal, -1, keepdim=True) * normal
    tangent = tangent / torch.linalg.norm(tangent, ord=2, dim=-1, keepdim=True)

    bitangent = torch.cross(normal, tangent, -1)

    direction = (
        direction[:, :, 0:1] * tangent[:, None, :]
        + direction[:, :, 1:2] * bitangent[:, None, :]
        + direction[:, :, 2:3] * normal[:, None, :]
    )
    direction = direction / torch.linalg.norm(direction, ord=2, dim=-1, keepdim=True)
    return direction


@torch.no_grad()
def sample_direction(N=1000, device="cpu", method="fibonacci"):
    """
    Uniformly sample directions (unit vectors) from the unit sphere.
    """
    match method:
        case "uniform":
            direction = torch.randn(N, 3, device=device)
        case "fibonacci":
            y = torch.linspace(-1, 1, N, device=device)
            radius = torch.sqrt(1 - y.square())
            theta = PHI * torch.arange(N, device=device)
            x = torch.cos(theta) * radius
            z = torch.sin(theta) * radius
            direction = torch.stack([x, y, z], dim=-1)
        case "grid":
            N = int(math.sqrt(N))
            arange = (0.5 + torch.arange(N, device=device)) / N
            u = arange[:, None].repeat(1, N)
            v = arange[None, :].repeat(N, 1)
            z = 1 - 2 * u
            phi = 2 * PI * v

            r = torch.sqrt(1 - z * z)
            x = r * torch.cos(phi)
            y = r * torch.sin(phi)

            direction = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        case "stratified_grid":
            N = int(math.sqrt(N))
            arange = torch.arange(N, device=device)
            u = (arange[:, None].repeat(1, N) + torch.rand(N, N, device=device)) / N
            v = (arange[None, :].repeat(N, 1) + torch.rand(N, N, device=device)) / N
            z = 1 - 2 * u
            phi = 2 * PI * v

            r = torch.sqrt(1 - z * z)
            x = r * torch.cos(phi)
            y = r * torch.sin(phi)

            direction = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

        case "jittered":
            fibonacci_dir = sample_direction(N=N, device=device, method="fibnoacci")
            uniform_dir = sample_direction(N=N, device=device, method="uniform")
            direction = fibonacci_dir + uniform_dir / N

    direction = direction / torch.linalg.norm(direction, ord=2, dim=-1, keepdim=True)
    return direction


@torch.no_grad()
def sample_from_patch(
    patch_vertex_coords, N=1000, is_triangle=True, method="equispaced"
):
    """
    Uniformly sample points from M triangles using barycentric coordinates.

    Parameters:
    patch_vertex_coords: torch.Tensor, shape [M, 3, 3], coordinates of the M triangle vertices
    num_sample: int, number of samples per triangle

    Returns:
    torch.Tensor, shape [M, num_sample, 3], sampled points from the triangles
    """
    M = patch_vertex_coords.shape[0]
    device = patch_vertex_coords.device

    # Generate random barycentric coordinates
    match method:
        case "uniform":
            st = torch.rand(2, M, N, 1, device=device)
        case "equispaced":
            arange = (0.5 + torch.arange(N, device=device)) / N
            s, t = torch.meshgrid(arange, arange, indexing="ij")
            st = torch.stack([s, t], dim=0).view(2, 1, N * N, 1)
        case "stratified":
            arange = (0.5 + torch.arange(N, device=device)) / N
            s, t = torch.meshgrid(arange, arange, indexing="ij")
            st = torch.stack([s, t], dim=0).view(2, 1, N * N, 1)
            noise = torch.rand(2, 1, N * N, 1, device=device)
            st = st + (2 * noise - 1) / (2 * N)
        case "r2":
            st = sample_quasirandom_r2(M * N, device=device)
            st = st.view(2, M, N, 1)
        case _:
            raise ValueError(f"Invalid method: {method}")

    # Adjust to ensure the points lie within the triangle
    if is_triangle:
        mask = st.sum(0) > 1.0
        st[:, mask] = 1.0 - st[:, mask]

    s, t = st

    x = patch_vertex_coords[:, 0:1]
    u = patch_vertex_coords[:, 1:2] - patch_vertex_coords[:, 0:1]
    v = patch_vertex_coords[:, -1:] - patch_vertex_coords[:, 0:1]

    # Compute the sampled points using barycentric interpolation
    sampled_points = x + s * u + t * v
    return sampled_points


@torch.no_grad()
def compute_local_orthonomal_matrix(patch_vertex_coords):
    x = patch_vertex_coords[:, 1] - patch_vertex_coords[:, 0]
    y = patch_vertex_coords[:, -1] - patch_vertex_coords[:, 0]
    x = x / torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    z = torch.cross(x, y, -1)
    z = z / torch.linalg.norm(z, ord=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, -1)
    R = torch.stack([x, y, z], dim=-1)
    return R


@torch.no_grad()
def discretize_direction(
    direction, N_ele, N_azi, bidirectional=True, local_orthonomal_matrix=None
):
    """
    Discretize directions using z/phi binning for approximately equal solid angle bins.

    Args:
        direction: [M, 3] tensor of unit direction vectors.
        N_ele: number of elevation-like bins (on z = cos(theta), not theta directly).
        N_azi: number of azimuthal bins (uniform in phi).
        bidirectional: if True, cover full sphere (z in [-1, 1]); else, hemisphere (z in [0, 1]).
        local_orthonomal_matrix: optional [M, 3, 3] matrices to rotate directions into local coordinates.

    Returns:
        direction_id: [M] flattened bin index (ele_bin * N_azi + azi_bin)

    Notes:
        - Although named `N_ele`, elevation binning is done over `z = cos(theta)` to better approximate
          equal solid angle per bin. Uniform binning in theta would distort area toward the poles.
    """

    if local_orthonomal_matrix is not None:
        # Rotate into local coordinate frame
        direction = torch.einsum("bji,bj->bi", local_orthonomal_matrix, direction)

    # Normalize just in case
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)

    # Get z = cos(theta)
    z = direction[:, 2].clamp(-1.0, 1.0)

    if not bidirectional:
        # Clamp negative z to hemisphere
        z = z.clamp(min=0.0)

    # Map z ∈ [z_min, 1] → [0, N_ele)
    z_min = -1.0 if bidirectional else 0.0
    ele_bin = ((z - z_min) / (1.0 - z_min) * N_ele).long()
    ele_bin = torch.clamp(ele_bin, max=N_ele - 1)
    # flip
    ele_bin = N_ele - 1 - ele_bin

    # Azimuth φ ∈ [0, 2π)
    phi = torch.atan2(direction[:, 1], direction[:, 0])
    phi = (phi + 2 * math.pi) % (2 * math.pi)
    azi_bin = (phi / (2 * math.pi) * N_azi).long()
    azi_bin = torch.clamp(azi_bin, max=N_azi - 1)

    # Flatten to single index
    direction_id = ele_bin * N_azi + azi_bin  # [M]

    return direction_id


@torch.no_grad()
def compute_bin_centers(N_ele, N_azi, bidirectional=True, device=None):
    """
    Computes the center direction (unit vectors) of each z/phi bin.

    Args:
        N_ele: number of elevation bins (z = cos(theta))
        N_azi: number of azimuth bins (phi = atan2)
        bidirectional: if True, z ∈ [-1, 1]; else z ∈ [0, 1] (hemisphere)
        device: optional torch device

    Returns:
        directions: [N_ele, N_azi, 3] tensor of unit vectors (x, y, z)
    """
    device = device or "cpu"

    # z range (cos(theta))
    z_min = -1.0 if bidirectional else 0.0
    z_max = 1.0
    dz = (z_max - z_min) / N_ele
    dphi = (2 * math.pi) / N_azi

    # Center of z-bins
    z_centers = torch.linspace(
        z_min + dz / 2, z_max - dz / 2, steps=N_ele, device=device
    )  # [N_ele]
    z_centers = torch.flip(z_centers, (-1,))
    phi_centers = torch.linspace(
        0 + dphi / 2, 2 * math.pi - dphi / 2, steps=N_azi, device=device
    )  # [N_azi]

    # Meshgrid to get all bin centers
    z_grid, phi_grid = torch.meshgrid(
        z_centers, phi_centers, indexing="ij"
    )  # [N_ele, N_azi]

    r = torch.sqrt(1 - z_grid**2).clamp(min=0.0)  # radial component in xy-plane
    x = r * torch.cos(phi_grid)
    y = r * torch.sin(phi_grid)
    z = z_grid

    directions = torch.stack([x, y, z], dim=-1)  # [N_ele, N_azi, 3]

    return directions  # unit vectors pointing in center of each bin


@torch.no_grad()
def compute_bin_solid_angles(N_ele, N_azi, bidirectional=True):
    z_min = -1.0 if bidirectional else 0.0
    z_max = 1.0
    dz = (z_max - z_min) / N_ele
    dphi = (2 * math.pi) / N_azi
    return dz * dphi
