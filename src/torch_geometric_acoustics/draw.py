r"""
Some utilities to draw room geometry mesh.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from torch_geometric_acoustics.core import compute_patch_geometric_properties


def draw_mesh_multiview(
    mesh=None,
    vertex_coords=None,
    patch_vertex_ids=None,
    views=["normal", "top", "front", "side"],
    **kwargs,
):
    fig, axes = plt.subplots(1, len(views), subplot_kw={"projection": "3d"})
    i = 0
    for view in views:
        match view:
            case "normal":
                pass
            case "top":
                axes[i].view_init(elev=90, azim=-90)
            case "front":
                axes[i].view_init(elev=0, azim=-90)
            case "side":
                axes[i].view_init(elev=0, azim=0)
        draw_mesh(mesh, vertex_coords, patch_vertex_ids, ax=axes[i], **kwargs)
        i += 1

    fig.subplots_adjust(wspace=0.1)
    fig.set_size_inches(10 * len(views), 10)
    return fig, axes


def draw_mesh_multiview_2(
    mesh=None,
    vertex_coords=None,
    patch_vertex_ids=None,
    **kwargs,
):
    fig, ax = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
    for i in range(2):
        for j in range(2):
            match (i, j):
                case (0, 0):
                    pass
                case (0, 1):
                    ax[i, j].view_init(elev=90, azim=-90)
                case (1, 0):
                    ax[i, j].view_init(elev=0, azim=-90)
                case (1, 1):
                    ax[i, j].view_init(elev=0, azim=0)
            draw_mesh(mesh, vertex_coords, patch_vertex_ids, ax=ax[0, 0], **kwargs)
            draw_mesh(mesh, vertex_coords, patch_vertex_ids, ax=ax[0, 1], **kwargs)
            draw_mesh(mesh, vertex_coords, patch_vertex_ids, ax=ax[1, 0], **kwargs)
            draw_mesh(mesh, vertex_coords, patch_vertex_ids, ax=ax[1, 1], **kwargs)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.set_size_inches(12, 12)
    return fig, ax


def draw_mesh_with_source_receiver(
    source_pos=None,
    receiver_pos=None,
    **mesh_kwargs,
):
    fig, axes = draw_mesh_multiview(**mesh_kwargs)
    if source_pos is not None:
        source_pos = source_pos.cpu().numpy()
        for ax in axes:
            ax.scatter(
                source_pos[:, 0],
                source_pos[:, 1],
                source_pos[:, 2],
                c="r",
                marker="o",
                label="source",
                s=20,
            )
    if receiver_pos is not None:
        receiver_pos = receiver_pos.cpu().numpy()
        for ax in axes:
            ax.scatter(
                receiver_pos[:, 0],
                receiver_pos[:, 1],
                receiver_pos[:, 2],
                c="b",
                marker="o",
                label="receiver",
                s=20,
            )
    return fig, ax


def draw_mesh(
    mesh=None,
    vertex_coords=None,
    patch_vertex_ids=None,
    ax=None,
    show_center=False,
    show_normal=False,
    show_vertex_id=False,
    show_patch_id=False,
    #   color_by="area",
    color_by="patch_idx",
    cmap="jet",
    alpha=0.15,
    show_axis=False,
    edgecolors="darkgray",
    linewidth=1,
):
    if mesh is not None:
        vertex_coords = torch.tensor(mesh.vertex_coords)
        patch_vertex_ids = torch.tensor(
            list(mesh.patch_vertex_ids.values()), dtype=torch.long
        )

    try:
        patch_vertex_coords, normal, area = compute_patch_geometric_properties(
            vertex_coords, patch_vertex_ids
        )
        center = patch_vertex_coords.mean(-2)
    except:
        patch_vertex_coords = vertex_coords[patch_vertex_ids]
    if ax is None:
        create = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        create = False
    cmap = plt.get_cmap(cmap)
    if isinstance(color_by, str):
        match color_by:
            case "area":
                if area.min() == area.max():
                    colors = "grey"
                else:
                    area_normalized = (area - area.min()) / (area.max() - area.min())
                    colors = cmap(area_normalized)

            case "patch_idx":
                arange = np.arange(len(patch_vertex_ids))
                colors = cmap(arange / len(patch_vertex_ids))

            case None:
                colors = "grey"

            case _:
                colors = color_by
    else:
        colors = color_by

    if show_center:
        ax.scatter(
            center[:, 0],
            center[:, 1],
            center[:, 2],
            c=colors,
            edgecolors="k",
            marker="o",
        )
    if show_normal:
        ax.quiver(
            center[:, 0],
            center[:, 1],
            center[:, 2],
            normal[:, 0],
            normal[:, 1],
            normal[:, 2],
            length=0.5,
            color=colors,
        )
    ax.add_collection3d(
        Poly3DCollection(
            patch_vertex_coords,
            facecolors=colors,
            linewidths=linewidth,
            edgecolors=edgecolors,
            alpha=alpha,
        )
    )
    if show_vertex_id:
        for i, v in enumerate(vertex_coords):
            ax.text(
                v[0],
                v[1],
                v[2],
                f"{i}",
                bbox=dict(facecolor="yellow", linewidth=0),
                zorder=10,
                fontsize=8,
            )
    if show_patch_id:
        for i, v in enumerate(center):
            ax.text(
                v[0],
                v[1],
                v[2],
                f"{i}",
                bbox=dict(facecolor="teal", linewidth=0),
                zorder=10,
                fontsize=8,
            )

    scale = patch_vertex_coords.flatten()
    v_min, v_max = vertex_coords.min(), vertex_coords.max()
    xs, ys, zs = np.array(vertex_coords).T
    ax.set_xlim(np.min(xs), np.max(xs))
    ax.set_ylim(np.min(ys), np.max(ys))
    ax.set_zlim(np.min(zs), np.max(zs))
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if create:
        fig.set_size_inches(10, 10)
        return fig, ax
    else:
        return ax


def draw_radiance_energy(
    vertex_coords,
    patch_vertex_ids,
    radiance_ids,
    radiance_energy,
    ax=None,
    show_center=True,
    show_normal=True,
    show_vertex_id=True,
    show_patch_id=False,
    #   color_by="area",
    color_by="patch_idx",
    cmap="jet",
    alpha=0.15,
    show_axis=False,
):
    patch_vertex_coords, normal, area = compute_patch_geometric_properties(
        vertex_coords, patch_vertex_ids
    )
    center = patch_vertex_coords.mean(-2)
    if ax is None:
        create = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        create = False
    cmap = plt.get_cmap(cmap)
    match color_by:
        case "area":
            if area.min() == area.max():
                colors = "grey"
            else:
                area_normalized = (area - area.min()) / (area.max() - area.min())
                colors = cmap(area_normalized)

        case "patch_idx":
            arange = np.arange(len(patch_vertex_ids))
            colors = cmap(arange / len(patch_vertex_ids))

        case None:
            colors = "grey"

    ax.add_collection3d(
        Poly3DCollection(
            patch_vertex_coords,
            facecolors=colors,
            linewidths=1,
            edgecolors="darkgray",
            alpha=alpha,
        )
    )
    ax.quiver(
        center[:, 0],
        center[:, 1],
        center[:, 2],
        normal[:, 0],
        normal[:, 1],
        normal[:, 2],
        length=0.5,
        color=colors,
    )

    scale = patch_vertex_coords.flatten()
    v_min, v_max = vertex_coords.min(), vertex_coords.max()
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(v_min, v_max)
    ax.set_zlim(v_min, v_max)
    ax.auto_scale_xyz(scale, scale, scale)
    if not show_axis:
        ax.set_axis_off()
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if create:
        fig.set_size_inches(10, 10)
        return fig, ax
    else:
        return ax


def draw_mesh_with_radiance(
    mesh,
    radiance,
    radiance_pos,
    radiance_dir,
    source_pos=None,
    receiver_pos=None,
    views=["normal", "top"],
    **draw_kwargs,
):
    radiance = radiance.cpu().numpy()
    radiance_pos = radiance_pos.cpu().numpy()
    radiance_dir = radiance_dir.cpu().numpy()
    if source_pos is not None:
        source_pos = source_pos.cpu().numpy()
    if receiver_pos is not None:
        receiver_pos = receiver_pos.cpu().numpy()

    radiance_energy = radiance.sum(-1)
    normalized_energy = radiance_energy / radiance_energy.max()

    scaled_radiance_direction = radiance_dir * normalized_energy[:, None]
    mask = normalized_energy > 0
    nonzero_radiance_pos = radiance_pos[mask]
    nonzero_radiance_dir = scaled_radiance_direction[mask]
    nonzero_radiance_energy = normalized_energy[mask]

    fig, axes = draw_mesh_multiview(mesh, views=views, **draw_kwargs)
    cmap = plt.get_cmap("jet")

    for ax in axes:
        ax.quiver(
            nonzero_radiance_pos[:, 0],
            nonzero_radiance_pos[:, 1],
            nonzero_radiance_pos[:, 2],
            nonzero_radiance_dir[:, 0],
            nonzero_radiance_dir[:, 1],
            nonzero_radiance_dir[:, 2],
            color=cmap(nonzero_radiance_energy),
        )
        if source_pos is not None:
            ax.scatter(
                source_pos[:, 0],
                source_pos[:, 1],
                source_pos[:, 2],
                c="r",
                marker="o",
                label="source",
                s=20,
            )
        if receiver_pos is not None:
            ax.scatter(
                receiver_pos[:, 0],
                receiver_pos[:, 1],
                receiver_pos[:, 2],
                c="b",
                marker="o",
                label="receiver",
                s=20,
            )
    return fig, axes
