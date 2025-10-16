"""
Code for direct arrival term.
"""
import torch

from torch_geometric_acoustics.core import delay_impulse, intersection_test


def compute_direct_component(
    source_pos,
    receiver_pos,
    patch_vertex_coords,
    radiance_sampling_rate,
    speed_of_sound=343,
    echogram_len=4096,
    source_directivity=None,
    source_orientation=None,
    receiver_directivity=None,
    receiver_orientation=None,
):
    device = source_pos.device

    intersect_mask, _ = intersection_test(
        patch_vertex_coords, source_pos, target=receiver_pos
    )
    any_intersect = intersect_mask.any()
    if any_intersect:
        direct_echogram = torch.zeros(echogram_len, device=device)
    else:
        source_to_receiver = receiver_pos - source_pos
        distance = torch.norm(source_to_receiver, dim=-1)
        delay = distance / speed_of_sound
        delay = delay * radiance_sampling_rate
        direct_echogram = delay_impulse(delay, echogram_len, method="fraction_linear")
        direct_echogram = direct_echogram / (distance.square() * 4 * torch.pi)
        if source_directivity is not None:
            source_to_receiver = source_to_receiver / distance[:, None]
            source_directivity = source_directivity(
                source_to_receiver, orientation=source_orientation
            )
            direct_echogram = direct_echogram * source_directivity
        direct_echogram = direct_echogram[0]

    return direct_echogram
