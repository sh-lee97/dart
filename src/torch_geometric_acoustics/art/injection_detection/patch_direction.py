"""
Injection & detection for patch-direction (PD) and patch-direction-factorized (PDF) ART.
"""

import torch
import torch.fft
from torch_geometric_acoustics.core import (
    delay_impulse,
    discretize_direction,
    find_first_intersection,
    sample_direction,
)
from torch_scatter import scatter
from torchaudio.functional import fftconvolve


def get_radiance_id(dense_radiance_id, valid_radiance_ids):
    patch_pair_mask = valid_radiance_ids[:, None, :] == dense_radiance_id[:, :, None]
    patch_pair_mask = patch_pair_mask.all(0)
    match_id, radiance_id = torch.nonzero(patch_pair_mask, as_tuple=True)
    return match_id, radiance_id


# @profile
def compute_inject_radiance(
    source_pos,
    patch_vertex_coords,
    valid_radiance_ids,
    geometry,
    injection_scattering_matrix,
    local_orthonomal_matrix,
    injection_residual_matrix=None,
    num_rays=1000,
    echogram_len=4096,
    radiance_sampling_rate=4096,
    speed_of_sound=343,
    source_orientation=None,
    source_directivity=None,
    sampling_method="grid",
    direction=None,
    return_minimum_delays=False,
    aggregate_delay=False,
    N_ele=4,
    N_azi=4,
    bidirectional=True,
):
    r"""
    Zero-th order injection with injection scattering matrix
    This is equivalent to first-order injection
    when the discretized radiances are identical for all input directions in the same bin.
    """
    device = patch_vertex_coords.device

    if direction is None:
        direction = sample_direction(N=num_rays, device=device, method=sampling_method)
    num_radiance = valid_radiance_ids.shape[1]

    source_pos = source_pos.repeat(num_rays, 1)
    num_radiances = len(geometry)

    with torch.no_grad():
        any_intersection, patch_id, _, distance = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=source_pos,
            direction=direction,
        )

        incident_orthonomal_matrix = local_orthonomal_matrix[patch_id]
        direction_id = discretize_direction(
            -direction.view(-1, 3),
            N_ele=N_ele,
            N_azi=N_azi,
            bidirectional=bidirectional,
            local_orthonomal_matrix=incident_orthonomal_matrix,
        )

        dense_radiance_id = torch.stack([patch_id, direction_id])
        match_id, radiance_id = get_radiance_id(dense_radiance_id, valid_radiance_ids)

        distance = distance[match_id]
        direction = direction[match_id]
        delays = distance * (radiance_sampling_rate / speed_of_sound)

        if not any_intersection.all():
            any_intersection = any_intersection[match_id]
            radiance_id = radiance_id[any_intersection]
            distance = distance[any_intersection]
            delays = delays[any_intersection]
            direction = direction[any_intersection]

    if source_directivity is None:
        ray_energy = torch.ones_like(delays, dtype=torch.float32)
        ray_energy = ray_energy * (geometry.mean() / num_rays)
    else:
        ray_energy = source_directivity(direction, orientation=source_orientation)

        ray_energy = ray_energy * (geometry.mean() / ray_energy.sum())

    if aggregate_delay:
        average_delay = scatter(
            delays, radiance_id, dim_size=num_radiance, reduce="mean"
        )
        aggregate_energy = scatter(
            ray_energy, radiance_id, dim_size=num_radiance, reduce="sum"
        )

        zero_order_power = delay_impulse(
            delay_samples=average_delay,
            signal_len=echogram_len,
            method="fraction_linear",
        )
        zero_order_power = zero_order_power * aggregate_energy[:, None]

    else:
        delays = delays.round().long()
        scatter_id = radiance_id * echogram_len + delays
        zero_order_power = scatter(
            ray_energy, scatter_id, dim_size=num_radiances * echogram_len, reduce="sum"
        )

    zero_order_power = zero_order_power.view(num_radiances, echogram_len)
    zero_order_radiance = zero_order_power / geometry[:, None]
    injected_radiance = injection_scattering_matrix.matmul(zero_order_radiance)

    if injection_residual_matrix is not None:
        injected_residual_radiance = injection_residual_matrix.matmul(
            zero_order_radiance
        )
    else:
        injected_residual_radiance = None

    if return_minimum_delays:
        min_delay = (injected_radiance > 0).float().argmax(-1)
        return injected_radiance, injected_residual_radiance, min_delay
    else:
        return injected_radiance, injected_residual_radiance


def detect_echogram(
    receiver_pos,
    radiance,
    patch_vertex_coords,
    valid_radiance_ids,
    geometry,
    local_orthonomal_matrix,
    num_rays=1000,
    echogram_len=4096,
    radiance_sampling_rate=4096,
    speed_of_sound=343,
    receiver_orientation=None,
    receiver_directivity=None,
    sampling_method="grid",
    direction=None,
    N_ele=4,
    N_azi=4,
    bidirectional=True,
):
    device = patch_vertex_coords.device

    if direction is None:
        direction = sample_direction(N=num_rays, device=device, method=sampling_method)
    num_radiance = valid_radiance_ids.shape[1]

    receiver_pos = receiver_pos.repeat(num_rays, 1)
    num_radiances = len(geometry)

    with torch.no_grad():
        any_intersection, patch_id, _, distance = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=receiver_pos,
            direction=direction,
        )

        incident_orthonomal_matrix = local_orthonomal_matrix[patch_id]
        direction_id = discretize_direction(
            -direction.view(-1, 3),
            N_ele=N_ele,
            N_azi=N_azi,
            bidirectional=bidirectional,
            local_orthonomal_matrix=incident_orthonomal_matrix,
        )

        dense_radiance_id = torch.stack([patch_id, direction_id])
        match_id, radiance_id = get_radiance_id(dense_radiance_id, valid_radiance_ids)

        distance = distance[match_id]
        delays = distance * (radiance_sampling_rate / speed_of_sound)

        if not any_intersection.all():
            any_intersection = any_intersection[match_id]
            radiance_id = radiance_id[any_intersection]
            distance = distance[any_intersection]
            delays = delays[any_intersection]

    if receiver_directivity is None:
        scale = 4 * torch.pi * 1 / num_rays
        if geometry is not None:
            scale = scale / geometry.mean()
        ray_energy = torch.ones(len(radiance_id), device=device) * scale
    else:
        ray_energy = receiver_directivity(receiver_orientation, direction)
        ray_energy = ray_energy / ray_energy.sum()  # * num_rays

    average_delay = scatter(delays, radiance_id, dim_size=num_radiance, reduce="mean")
    aggregate_energy = scatter(
        ray_energy, radiance_id, dim_size=num_radiance, reduce="sum"
    )

    detection_ir = delay_impulse(
        delay_samples=average_delay, signal_len=echogram_len, method="fraction_linear"
    )
    detection_ir = detection_ir * aggregate_energy[:, None]

    if isinstance(radiance, list):
        echogram = []
        for i in range(len(radiance)):
            echogram_i = fftconvolve(radiance[i], detection_ir, mode="full")
            echogram_i = echogram_i.sum(-2)
            echogram_i = echogram_i[:echogram_len]
            echogram.append(echogram_i)
    else:
        echogram = fftconvolve(radiance, detection_ir, mode="full")
        echogram = echogram.sum(-2)
        echogram = echogram[:echogram_len]

    return echogram
