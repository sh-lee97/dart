r"""
Some plot tools.
"""
import matplotlib
import matplotlib.pyplot as plt
import torch

from torch_geometric_acoustics.draw import draw_mesh

plt.style.use("seaborn-v0_8-ticks")
plt.style.use("tableau-colorblind10")
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.linewidth"] = 0.8


@torch.no_grad()
def compare_echogram(gt_echogram, pred_echogram, path=None):
    gt_echogram = gt_echogram.detach().cpu().numpy()
    pred_echogram = pred_echogram.detach().cpu().numpy()
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(gt_echogram, "k", label="gt")
    ax[0].plot(pred_echogram, "b", label="pred")
    ax[0].legend()
    ax[0].set_yscale("symlog", linthresh=1e-7)
    ax[1].plot(gt_echogram, "k", label="gt")
    ax[1].plot(pred_echogram, "b", label="pred")
    ax[1].legend()
    fig.set_size_inches(12, 8)
    if path is None:
        path = "experiments/outputs/consistency_check_echogram.pdf"
    fig.savefig(path)
    plt.close(fig)

@torch.no_grad()
def plot_material_coefficients(mesh, absorption_coefficient, path=None):
    absorption_coefficient = absorption_coefficient.detach().cpu().numpy()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex=True, sharey=True)
    color_by = absorption_coefficient
    cmap = plt.get_cmap("jet")
    color_by = cmap(color_by)
    draw_mesh(
        mesh=mesh,
        ax=ax,
        color_by=color_by,
        show_center=False,
        show_normal=False,
        show_vertex_id=False,
    )
    fig.set_size_inches(12, 5)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    if path is None:
        path = "experiments/outputs/consistency_check_material_coefficients.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap("jet")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots()
    ax.axis("off")
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal")
    fig.set_size_inches(3, 3)
    cbar.set_label(r"$\alpha$")  # optional label
    fig.savefig(path.replace(".pdf", ".cbar.pdf"), bbox_inches="tight", pad_inches=0.1)