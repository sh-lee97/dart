"""
Injection & detection for patch-to-patch (P2P) ART.
"""

import torch
import torch.fft
from torch_scatter import scatter
from torchaudio.functional import fftconvolve

from torch_geometric_acoustics.core import intersection_test, sample_direction



def get_radiance_id(dense_radiance_id, valid_radiance_ids):
    patch_pair_mask = valid_radiance_ids[:, None, :] == dense_radiance_id[:, :, None]
    patch_pair_mask = patch_pair_mask.all(0)
    match_id, radiance_id = torch.nonzero(patch_pair_mask, as_tuple=True)
    return match_id, radiance_id


def compute_inject_radiance_with_scattering_matrix(
    source_pos,
    patch_vertex_coords,
    valid_radiance_ids,
    scattering_matrix,
    geometry,
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
):
    device = patch_vertex_coords.device

    if direction is None:
        direction = sample_direction(N=num_rays, device=device, method=sampling_method)
    num_radiance = valid_radiance_ids.shape[1]

    source_pos = source_pos.repeat(num_rays, 1)
    num_radiances = len(geometry)

    with torch.no_grad():
        intersection, distance = intersection_test(
            patch_vertex_coords=patch_vertex_coords,
            origin=source_pos,
            direction=direction,
            bidirectional=True,
        )

        dest_patch_intersection = intersection * (distance > 0)
        source_patch_intersection = intersection * (distance < 0)
        source_distance = distance.clone()
        source_distance[~source_patch_intersection] = float("-inf")

        ray_id, dest_patch_id = torch.nonzero(dest_patch_intersection, as_tuple=True)
        distance = distance[dest_patch_intersection]

        source_patch_id = source_distance.argmax(-1)
        source_any_intersection = source_patch_intersection.any(-1)
        source_patch_id = source_patch_id[ray_id]

        dense_radiance_id = torch.stack([source_patch_id, dest_patch_id])
        match_id, radiance_id = get_radiance_id(dense_radiance_id, valid_radiance_ids)

        distance = distance[match_id]
        delays = distance * (radiance_sampling_rate / speed_of_sound)

        if not source_any_intersection.all():
            source_any_intersection = source_any_intersection[ray_id][match_id]
            radiance_id = radiance_id[source_any_intersection]
            distance = distance[source_any_intersection]
            delays = delays[source_any_intersection]

    if source_directivity is None:
        ray_energy = torch.ones_like(delays, dtype=torch.float32)
        ray_energy = ray_energy * (geometry.mean() / num_rays)
    else:
        direction = direction[match_id]
        ray_energy = source_directivity(source_orientation, direction)
        ray_energy = ray_energy / ray_energy.sum()  # * num_rays

    if aggregate_delay:
        average_delay = scatter(
            delays, radiance_id, dim_size=num_radiance, reduce="mean"
        )
        average_delay = average_delay.round().long()
        aggregate_energy = scatter(
            ray_energy, radiance_id, dim_size=num_radiance, reduce="sum"
        )

        zero_order_power = torch.zeros(num_radiance, echogram_len, device=device)
        zero_order_power[torch.arange(num_radiance), average_delay] = aggregate_energy
    else:
        delays = delays.round().long()
        scatter_id = radiance_id * echogram_len + delays
        zero_order_power = scatter(
            ray_energy, scatter_id, dim_size=num_radiances * echogram_len, reduce="sum"
        )

    zero_order_power = zero_order_power.view(num_radiances, echogram_len)
    zero_order_radiance = zero_order_power / geometry[:, None]
    injected_radiance = scattering_matrix.matmul(zero_order_radiance)

    if return_minimum_delays:
        min_delay = (injected_radiance > 0).float().argmax(-1)
        return injected_radiance, min_delay
    else:
        return injected_radiance


def detect_echogram(
    receiver_pos,
    radiance,
    patch_vertex_coords,
    valid_radiance_ids,
    num_rays=1000,
    echogram_len=4096,
    radiance_sampling_rate=4096,
    speed_of_sound=343,
    receiver_orientation=None,
    receiver_directivity=None,
    sampling_method="stratified_grid",
    direction=None,
    geometry=None,
):
    device = patch_vertex_coords.device
    num_radiances = valid_radiance_ids.shape[1]

    if direction is None:
        direction = sample_direction(N=num_rays, device=device, method=sampling_method)
    receiver_pos = receiver_pos.repeat(num_rays, 1)

    if isinstance(radiance, list):
        num_radiance = len(radiance[0])
    else:
        num_radiance = len(radiance)

    with torch.no_grad():
        intersection, distance = intersection_test(
            patch_vertex_coords=patch_vertex_coords,
            origin=receiver_pos,
            direction=direction,
            bidirectional=True,
        )

        source_patch_intersection = intersection * (distance > 0)
        dest_patch_intersection = intersection * (distance < 0)

        source_distance = distance.clone()
        source_distance[~source_patch_intersection] = float("inf")
        source_distance, source_patch_id = source_distance.min(-1)
        source_any_intersection = source_patch_intersection.any(-1)

        dest_distance = distance.clone()
        dest_distance[~dest_patch_intersection] = float("-inf")
        dest_patch_id = dest_distance.argmax(-1)
        dest_any_intersection = dest_patch_intersection.any(-1)

        any_interaction = dest_any_intersection & source_any_intersection

        if not any_interaction.all():
            dest_patch_id = dest_patch_id[any_interaction]
            source_patch_id = source_patch_id[any_interaction]
            source_distance = source_distance[any_interaction]

        dense_radiance_id = torch.stack([source_patch_id, dest_patch_id])
        match_id, radiance_id = get_radiance_id(dense_radiance_id, valid_radiance_ids)
        source_distance = source_distance[match_id]

    if receiver_directivity is None:
        scale = 4 * torch.pi * 1 / num_rays
        if geometry is not None:
            scale = scale / geometry.mean()
        ray_energy = torch.ones(len(radiance_id), device=device) * scale
    else:
        ray_energy = receiver_directivity(receiver_orientation, direction)

    aggregate_energy = scatter(
        ray_energy, radiance_id, dim_size=num_radiance, reduce="sum"
    )

    average_distance = scatter(
        source_distance, radiance_id, dim_size=num_radiance, reduce="mean"
    )

    delay = average_distance * radiance_sampling_rate / speed_of_sound
    delay = delay.round().long()

    detection_ir = torch.zeros(num_radiance, echogram_len, device=device)
    detection_ir[torch.arange(num_radiance), delay] = aggregate_energy

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
