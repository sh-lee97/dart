r"""
Efficient GPU-enabled image-source method (ISM).
Note: we do not use this method by default in our NeurIPS submission.
It was initially developed to accelerate the image source search of DiffRIR.
The acceleration is done by checking the field-of-view and terminating early with the search tree.
It can find image sources up to order 7 or 8 several orders of magnitude faster than the DiffRIR.
It is possible to use ISM as standalone module, or instead use it along with ray tracing or DART.
Investigation of such hybrids is left as future work.
"""

import torch
from torch_geometric_acoustics.core import (
    delay_impulse,
    find_first_intersection,
    reflect_direction,
)
from torch_scatter import segment_csr


# @profile
def image_source_method(
    patch_vertex_coords,
    normal,
    source_pos,
    receiver_pos,
    max_order=2,
    check_unique_planes=True,
    check_fov=True,
    distance_threshold=340,
):
    r"""

    Args:
        patch_vertex_coords (:python:`FloatTensor`, (N, 3)): Vertex coordinates.
        patch_vertex_ids: (M, 3, 3), LongTensor
        source_pos: (3), FloatTensor
        receiver_pos: (3), FloatTensor

    Returns:
        image_source_pos: (n_images, max_order, 3), FloatTensor
        orders: (n_images), LongTensor
        reflection_ids: (n_images, max_order), FloatTensor
        reflection_mask: (n_images, max_order), BoolTensor
        distance: (n_images), FloatTensor
    """

    source_pos = source_pos + torch.randn_like(source_pos) * 1e-3
    device = patch_vertex_coords.device

    d = (-normal * patch_vertex_coords[:, 0]).sum(-1)
    if check_unique_planes:
        plane_normal, plane_d, plane_ids, plane_num_patches = find_unique_planes(
            normal, d
        )
    else:
        plane_normal = normal
        plane_d = d
    plane_normal_T = plane_normal.T
    d = d[None, :]

    source_pos = torch.atleast_2d(source_pos)
    num_planes = len(plane_normal)
    # print("num_planes", num_planes)
    plane_arange = torch.arange(num_planes, device=device)

    source_images = source_pos
    reflection_id_list = []
    visible_id_list = []

    source_images_list = []
    source_images_valid_list = []
    for order in range(1, max_order + 1):
        distance = torch.addmm(plane_d, source_images, plane_normal_T)
        mask = distance > 0

        if order == 1:
            reflection_ids = plane_arange[None, :, None]
        else:
            reflection_ids = reflection_ids.unsqueeze(1).repeat(1, num_planes, 1)
            arange = plane_arange[None, :, None].repeat(reflection_ids.shape[0], 1, 1)
            reflection_ids = torch.cat([reflection_ids, arange], dim=-1)

            if check_fov:
                if check_unique_planes:
                    fov_mask = compute_fov_mask_with_unique_planes(
                        source_images,
                        patch_vertex_coords,
                        reflection_ids,
                        plane_normal,
                        plane_d,
                        plane_ids,
                        plane_num_patches,
                    )

                else:
                    fov_mask = compute_fov_mask(
                        source_images,
                        patch_vertex_coords,
                        normal,
                        d,
                        reflection_ids,
                    )
                mask = mask & fov_mask

        candidate_images = (
            source_images[:, None] - 2 * plane_normal[None, :] * distance[:, :, None]
        )

        # masking
        id_image, id_patch = torch.where(mask)
        reflection_ids = reflection_ids[id_image, id_patch]
        candidate_images = candidate_images[id_image, id_patch]
        source_images = candidate_images
        reflection_id_list.append(reflection_ids)
        if check_unique_planes:
            (
                reflection_ids_valid,
                reflected_patch_ids,
                source_images_valid,
                travel_distance,
            ) = check_validity_with_unique_plane(
                reflection_ids,
                patch_vertex_coords,
                source_images,
                receiver_pos,
                plane_ids,
                plane_normal,
            )
            # print("reflection_ids_valid", reflection_ids_valid.shape)
        else:
            (
                reflection_ids_valid,
                reflected_patch_ids,
                source_images_valid,
                travel_distance,
            ) = check_validity(
                reflection_ids,
                patch_vertex_coords,
                source_images,
                receiver_pos,
                plane_normal,
            )

        source_images_list.append(source_images)
        source_images_valid_list.append(source_images_valid)

    return source_images_list, source_images_valid_list, reflection_ids_valid


# @profile
def find_unique_planes(normal, d):
    device = normal.device
    plane_parameters = torch.cat([normal, d[:, None]], dim=-1)
    plane, idx = torch.unique(plane_parameters, dim=0, return_inverse=True)
    normal, d = plane[:, :-1], plane[:, -1]
    num_planes = len(normal)
    mask = torch.arange(num_planes, device=device)[:, None] == idx[None, :]
    num_patches = mask.count_nonzero(-1)
    return normal, d, idx, num_patches


def compute_fov_mask(
    source_images, patch_vertex_coords, patch_normal, patch_d, reflection_ids
):
    source_images = source_images[:, None, None, :]
    last_id, this_id = reflection_ids[..., -2], reflection_ids[..., -1]
    last_patch_coords, this_patch_coords = (
        patch_vertex_coords[last_id],
        patch_vertex_coords[this_id],
    )
    last_patch_coords_roll = torch.roll(last_patch_coords, -1, dims=2)
    normal = torch.cross(
        last_patch_coords - source_images,
        last_patch_coords_roll - source_images,
        -1,
    )
    d = (-normal * source_images).sum(-1)
    distance = (normal[:, :, :, None, :] * this_patch_coords[:, :, None, :, :]).sum(-1)
    distance = distance + d[:, :, :, None]
    mask = (distance < 0).all(-1).any(-1)
    return mask


# @profile
def compute_fov_mask_with_unique_planes(
    source_images,
    patch_vertex_coords,
    reflection_ids,
    plane_normal,
    plane_d,
    patch_to_plane_ids,
    plane_num_patches,
    fov_criteria="patch",
):
    r"""
    source_images: (num_images, 3)
    patch_vertex_coords: (num_patch, 3, 3)
    reflection_ids: (num_images, num_planes, max_order)
    patch_to_plane_ids: (num_patch)
    """
    num_sources, num_planes, _ = reflection_ids.shape
    num_patch = len(patch_vertex_coords)

    patch_arange = torch.arange(num_patch, device=source_images.device)

    # LAST PLANE
    last_plane_id = reflection_ids[..., -2].view(-1)
    last_mask = patch_to_plane_ids[None, :] == last_plane_id[:, None]
    last_patch_arange = patch_arange[None, :].expand(len(last_plane_id), -1)
    last_patch_id = last_patch_arange[last_mask]
    last_mask_true = plane_num_patches[last_plane_id]

    # THIS PLANE
    this_plane_id = reflection_ids[..., -1].view(-1)
    this_mask_true_ = (
        patch_to_plane_ids[None, :] == this_plane_id[:, None]
    ).count_nonzero(-1)
    this_plane_id = this_plane_id.repeat_interleave(last_mask_true)
    this_mask = patch_to_plane_ids[None, :] == this_plane_id[:, None]
    this_patch_arange = patch_arange[None, :].expand(len(this_plane_id), -1)
    this_patch_id = this_patch_arange[this_mask]
    this_mask_true = plane_num_patches[this_plane_id]

    # PREPARE PATCH & SOURCE COORDS
    last_patch_id = last_patch_id.repeat_interleave(this_mask_true)
    last_patch_coords = patch_vertex_coords[last_patch_id]
    this_patch_coords = patch_vertex_coords[this_patch_id]
    repeats = last_mask_true[::num_planes] * num_patch
    source_images = source_images.repeat_interleave(repeats, dim=0)
    source_images = source_images[:, None, :]

    # CHECK IF IN FIELD FOR PATCH PAIRS
    last_patch_coords_roll = torch.roll(last_patch_coords, -1, dims=-2)
    source_to_coord = last_patch_coords - source_images
    source_to_coord_roll = last_patch_coords_roll - source_images
    normal = torch.cross(source_to_coord, source_to_coord_roll, -1)
    d = (-normal * source_images).sum(-1)

    # PLANE CHECK
    plane_param = torch.cat([plane_normal, plane_d[:, None]], dim=-1)
    last_plane_param = plane_param[patch_to_plane_ids[last_patch_id]]
    last_plane_normal, last_plane_d = (
        last_plane_param[:, :-1],
        last_plane_param[:, -1],
    )

    match fov_criteria:
        case "patch":
            distance = (normal[:, :, None, :] * this_patch_coords[:, None, :, :]).sum(
                -1
            )
            distance = distance + d[:, :, None]
            distance_plane = (last_plane_normal[:, None, :] * this_patch_coords).sum(
                -1
            ) + last_plane_d[:, None]
            out_of_field = (distance < 0).all(-1).any(-1) | (distance_plane < 0).all(-1)
        case "center":
            center = this_patch_coords.mean(-2)
            distance = (normal * center[:, None, :]).sum(-1) + d
            distance_plane = (last_plane_normal * center).sum(-1) + last_plane_d
            out_of_field = (distance < 0).any(-1) | (distance_plane < 0)

    # CHECK IF IN FIELD FOR PLANE PAIRS
    zeros_idx = (last_mask_true * this_mask_true_).cumsum(-1)
    zeros_idx = torch.cat([zeros_idx.new_zeros(1), zeros_idx])
    in_field_any = ~segment_csr(out_of_field.long(), zeros_idx, reduce="min").bool()
    in_field_any = in_field_any.view(num_sources, num_planes)
    return in_field_any


def check_validity_with_unique_plane(
    reflection_ids,
    patch_vertex_coords,
    source_images,
    receiver_pos,
    plane_ids,
    plane_normal,
):
    r"""
    Validity check of the image sources found with unique planes.
    From the given image sources and their (supposed) reflection *plane* ids,
    we check if the reflections actually happen on the correct *planes*.
    """
    receiver_pos = torch.atleast_2d(receiver_pos)
    receiver_pos = receiver_pos.repeat(len(source_images), 1)
    order = reflection_ids.shape[-1]

    ts = []
    reflected_patch_ids = []
    direction = source_images - receiver_pos
    for i in range(1, order + 1):
        reflection_ids_i = reflection_ids[..., -i]
        _, reflected_patch_id, receiver_pos, t = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=receiver_pos,
            direction=direction,
        )
        reflected_patch_ids.append(reflected_patch_id)
        reflected_plane_id = plane_ids[reflected_patch_id]
        direction = reflect_direction(plane_normal[reflected_plane_id], direction)
        if i == 1:
            valid = reflected_plane_id == reflection_ids_i
        else:
            valid = valid & (reflected_plane_id == reflection_ids_i)
        ts.append(t)

    ts = torch.stack(ts, dim=-1)
    reflection_ids_valid = reflection_ids[valid]
    reflected_patch_ids = torch.stack(reflected_patch_ids, dim=-1)
    reflected_patch_ids = reflected_patch_ids[valid]
    source_images_valid = source_images[valid]
    travel_distance = ts[valid]
    return (
        reflection_ids_valid,
        reflected_patch_ids,
        source_images_valid,
        travel_distance,
    )


def check_validity(
    reflection_ids,
    patch_vertex_coords,
    source_images,
    receiver_pos,
    plane_normal,
    gauranteed_intersection=False,
):
    r"""
    The simplest validity check.
    From the given image sources and their (supposed) reflection *patch* ids,
    we check if the reflections actually happen on the correct *patches*.
    """
    receiver_pos = torch.atleast_2d(receiver_pos)
    receiver_pos = receiver_pos.repeat(len(source_images), 1)
    order = reflection_ids.shape[-1]

    ts = []
    reflected_patch_ids = []
    direction = source_images - receiver_pos
    for i in range(1, order + 1):
        reflection_ids_i = reflection_ids[..., -i]
        any_intersection, reflected_patch_id, receiver_pos, t = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=receiver_pos,
            direction=direction,
        )
        direction = reflect_direction(plane_normal[reflected_patch_id], direction)
        if i == 1:
            valid = reflected_patch_id == reflection_ids_i
        else:
            valid = valid & (reflected_patch_id == reflection_ids_i)
        reflected_patch_ids.append(reflected_patch_id)
        ts.append(t)

    ts = torch.stack(ts, dim=-1)
    reflection_ids_valid = reflection_ids[valid]
    reflected_patch_ids = torch.stack(reflected_patch_ids, dim=-1)
    reflected_patch_ids = reflected_patch_ids[valid]
    source_images_valid = source_images[valid]
    travel_distance = ts[valid]
    return (
        reflection_ids_valid,
        reflected_patch_ids,
        source_images_valid,
        travel_distance,
    )


def get_reflection_ids(
    patch_vertex_coords,
    source_images,
    receiver_pos,
    plane_normal,
    order=5,
):
    r"""
    The simplest validity check.
    From the given image sources and their (supposed) reflection *patch* ids,
    we check if the reflections actually happen on the correct *patches*.
    """
    receiver_pos = torch.atleast_2d(receiver_pos)
    receiver_pos = receiver_pos.repeat(len(source_images), 1)

    ts = []
    reflected_patch_ids = []
    direction = source_images - receiver_pos
    for i in range(order):
        any_intersection, reflected_patch_id, receiver_pos, t = find_first_intersection(
            patch_vertex_coords=patch_vertex_coords,
            origin=receiver_pos,
            direction=direction,
        )
        direction = reflect_direction(plane_normal[reflected_patch_id], direction)
        if i == 0:
            valid = any_intersection
        else:
            valid = valid & any_intersection
        reflected_patch_ids.append(reflected_patch_id)
        ts.append(t)

    reflected_patch_ids = torch.stack(reflected_patch_ids, dim=-1)
    reflected_patch_ids = reflected_patch_ids[valid]
    return reflected_patch_ids


def single_order_ism(
    patch_vertex_coords,
    normal,
    source_pos,
    receiver_pos,
    radiance_sampling_rate=1000,
    echogram_len=320,
    speed_of_sound=343,
    source_directivity=None,
    source_orientation=None,
    absorption_coefficient=None,
    scattering_coefficient=None,
):
    # flip the source & receiver
    # s.t. we do not need to check the orientation of the source images
    _, receiver_images, reflection_ids_valid = image_source_method(
        patch_vertex_coords,
        normal,
        source_pos=receiver_pos[0],
        receiver_pos=source_pos[0],
        max_order=1,
        check_unique_planes=False,
        check_fov=False,
    )

    receiver_images = receiver_images[0]
    reflection_ids_valid = reflection_ids_valid[:, 0]
    if len(receiver_images) != 0:
        directions = receiver_images - source_pos
        distances = directions.norm(dim=-1)

        absorption_coefficient = absorption_coefficient[reflection_ids_valid]
        specular_weight = scattering_coefficient[0, reflection_ids_valid]

        delays = distances * (radiance_sampling_rate / speed_of_sound)
        delayed_signal = delay_impulse(
            delay_samples=delays, signal_len=echogram_len, method="fraction_linear"
        )
        amp = torch.ones_like(distances) / (distances.square() * 4 * torch.pi)
        amp = amp * absorption_coefficient
        amp = amp * specular_weight
        if source_directivity is not None:
            directions = directions / distances[:, None]
            amp = amp * source_directivity(directions, orientation=source_orientation)
        delayed_signal = delayed_signal * amp[:, None]
        ism = delayed_signal.sum(0)
    else:
        ism = torch.zeros(echogram_len, device=source_pos.device)
    return ism
