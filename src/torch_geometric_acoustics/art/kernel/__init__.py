from torch_geometric_acoustics.art.kernel.patch_direction import (
    reflection_kernel_patch_direction,
)
from torch_geometric_acoustics.art.kernel.patch_direction_factorized import (
    reflection_kernel_patch_direction_factorized,
)
from torch_geometric_acoustics.art.kernel.patch_to_patch import (
    reflection_kernel_patch_to_patch,
)


def compute_reflection_kernel_basis(
    patch_vertex_coords,
    normal,
    brdfs=["diffuse", "specular"],
    method="patch-to-patch",
    **kwargs,
):
    r"""
    Compute the reflection kernel for each given BRDF.
    """
    match method:
        case "patch-to-patch":
            kernel = reflection_kernel_patch_to_patch(
                patch_vertex_coords,
                normal,
                brdfs=brdfs,
                **kwargs,
            )
        case "patch-direction":
            kernel = reflection_kernel_patch_direction(
                patch_vertex_coords,
                normal,
                brdfs=brdfs,
                **kwargs,
            )
        case "patch-direction-factorized":
            kernel = reflection_kernel_patch_direction_factorized(
                patch_vertex_coords,
                normal,
                brdfs=brdfs,
                **kwargs,
            )
    return kernel
