"""
Differentiable Acoustic Radiance Transfer (DART), a patch-to-patch (P2P) variant (discussed in Appendix).
Currently supports parametric variant with four BSDFs:
 - Ideal specular reflection
 - Ideal diffuse reflection
 - Ideal specular transmission
 - Ideal diffuse transmission
"""
import itertools
from functools import partial

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from torch_geometric_acoustics.art.direct import compute_direct_component
from torch_geometric_acoustics.art.geometry import integrated_geometry_patch_to_patch
from torch_geometric_acoustics.art.injection_detection.patch_to_patch import (
    compute_inject_radiance_with_scattering_matrix,
    detect_echogram,
)
from torch_geometric_acoustics.art.kernel import compute_reflection_kernel_basis
from torch_geometric_acoustics.art.kernel.sparse import (
    compile_basis_kernels,
    postprocess_basis_kernels,
)
from torch_geometric_acoustics.art.main_loop import (
    main_art_loop_sparse_mm,
    main_art_loop_sparse_mm_exact_delay,
)
from torch_geometric_acoustics.core import (
    compute_patch_geometric_properties,
    delay_impulse,
    sample_direction,
)


class AcousticRadianceTransfer_PatchToPatch(nn.Module):
    def __init__(
        self,
        mesh,
        radiance_sampling_rate=16000,
        speed_of_sound=343,
        echogram_len_sec=0.3,
        brdfs=["diffuse", "specular"],
        fsm_gamma=1e-3,
        num_bounces=40,
        num_injection_rays=10000,
        num_detection_rays=10000,
        direction_sampling_method="grid",
        main_loop_domain="frequency",
        air_absoprtion_coefficient=1e-3,
        direct_arrival=True,
        learnable_envelope=True,
    ):
        super().__init__()

        # --------------------------------
        # mesh tensors
        vertex_coords = torch.tensor(mesh.vertex_coords, dtype=torch.float32)
        patch_vertex_ids = torch.tensor(
            list(mesh.patch_vertex_ids.values()), dtype=torch.long
        )
        patch_vertex_coords, normal, area = compute_patch_geometric_properties(
            vertex_coords, patch_vertex_ids
        )
        self.register_buffer("patch_vertex_coords", patch_vertex_coords)
        self.register_buffer("normal", normal)
        self.register_buffer("area", area)

        # --------------------------------
        # parameters
        self.num_patches = len(patch_vertex_coords)
        self.fsm_gamma = fsm_gamma
        self.echogram_len = int(radiance_sampling_rate * echogram_len_sec)
        self.num_bounces = num_bounces
        self.radiance_sampling_rate = radiance_sampling_rate
        self.speed_of_sound = speed_of_sound
        self.brdfs = brdfs
        self.num_brdfs = len(brdfs)
        self.num_injection_rays = num_injection_rays
        self.num_detection_rays = num_detection_rays
        self.direction_sampling_method = direction_sampling_method
        self.main_loop_domain = main_loop_domain
        self.fsm_correection = fsm_gamma != 1 and main_loop_domain == "frequency"
        self.air_absoprtion_coefficient = air_absoprtion_coefficient

        # ---------------------------------
        # frequency sampling method (fsm)
        log_gamma = np.log(self.fsm_gamma)
        fsm_window = torch.arange(self.echogram_len) / self.radiance_sampling_rate
        fsm_window = torch.exp(log_gamma * fsm_window)
        fsm_window = fsm_window[None, :]
        self.register_buffer("fsm_window", fsm_window)

        if self.direction_sampling_method == "fibonacci":
            injection_direction = sample_direction(
                N=self.num_injection_rays, method="fibonacci", device="cuda"
            )
            self.register_buffer("injection_direction", injection_direction)
            detection_direction = sample_direction(
                N=self.num_detection_rays, method="fibonacci", device="cuda"
            )
            self.register_buffer("detection_direction", detection_direction)
        else:
            self.injection_direction = None
            self.detection_direction = None

        self.direct_arrival = direct_arrival
        self.learnable_envelope = learnable_envelope
        if self.learnable_envelope:
            self.envelope = nn.Parameter(
                torch.zeros(self.echogram_len, dtype=torch.float32)
            )

    @torch.no_grad()
    def precompute(self):
        print("Precomputing...")

        free, total = torch.cuda.mem_get_info()
        mem_used_MB_before = (total - free) / 1024**2

        # --------------------------------
        # geometry & kernel
        geometry = integrated_geometry_patch_to_patch(
            self.patch_vertex_coords, self.normal, self.area
        )
        kernel_basis, average_distance = compute_reflection_kernel_basis(
            patch_vertex_coords=self.patch_vertex_coords,
            normal=self.normal,
            brdfs=self.brdfs,
        )

        # --------------------------------
        # sparse
        radiance_mask = geometry > 0  # SURFACE BOTH SIDES ??? HANDLE HERE
        kernel_basis, nonzero_radiance_mask = postprocess_basis_kernels(
            kernels=kernel_basis,
            radiance_mask=radiance_mask,
        )

        geometry = geometry[nonzero_radiance_mask]
        average_distance = average_distance[nonzero_radiance_mask]

        valid_radiance_ids = list(itertools.product(range(self.num_patches), repeat=2))
        valid_radiance_ids = torch.tensor(valid_radiance_ids, device="cuda")
        valid_radiance_ids = valid_radiance_ids[nonzero_radiance_mask].T

        self.num_radiances = valid_radiance_ids.shape[1]

        free, total = torch.cuda.mem_get_info()
        mem_used_MB_after = (total - free) / 1024**2
        torch.cuda.empty_cache()
        print("Precomputing... Done")
        print(
            f"Used (additional) memory: {mem_used_MB_after - mem_used_MB_before:.2f} MB"
        )
        row, col, sparse_kernel_basis = compile_basis_kernels(kernel_basis, self.brdfs)
        # source patch of the output radiance
        reflector_ids = valid_radiance_ids[0][row]

        self.register_buffer("geometry", geometry)
        self.register_buffer("valid_radiance_ids", valid_radiance_ids)
        self.register_buffer("sparse_kernel_row", row)
        self.register_buffer("sparse_kernel_col", col)
        self.register_buffer("sparse_kernel_basis", sparse_kernel_basis)
        self.register_buffer("sparse_kernel_reflector_id", reflector_ids)

        # ---------------------------------
        # delay
        delay_samples = average_distance * (
            self.radiance_sampling_rate / self.speed_of_sound
        )
        delay_signal = delay_impulse(
            delay_samples=delay_samples,
            signal_len=self.echogram_len,
            method="integer_nearest",
        )
        delay_samples = delay_samples.round().long()
        self.register_buffer("delay_signal", delay_signal)
        self.register_buffer("delay_samples", delay_samples)

        self.injection = partial(
            compute_inject_radiance_with_scattering_matrix,
            patch_vertex_coords=self.patch_vertex_coords,
            valid_radiance_ids=self.valid_radiance_ids,
            geometry=self.geometry,
            num_rays=self.num_injection_rays,
            echogram_len=self.echogram_len,
            radiance_sampling_rate=self.radiance_sampling_rate,
            speed_of_sound=self.speed_of_sound,
            sampling_method=self.direction_sampling_method,
            aggregate_delay=True,
        )

        self.detection = partial(
            detect_echogram,
            patch_vertex_coords=self.patch_vertex_coords,
            valid_radiance_ids=self.valid_radiance_ids,
            num_rays=self.num_detection_rays,
            echogram_len=self.echogram_len,
            radiance_sampling_rate=self.radiance_sampling_rate,
            speed_of_sound=self.speed_of_sound,
            sampling_method=self.direction_sampling_method,
            geometry=self.geometry,
        )

        # --------------------------------
        # memory
        free, total = torch.cuda.mem_get_info()
        mem_used_MB_after = (total - free) / 1024**2
        torch.cuda.empty_cache()
        print("Precomputing... Done")
        print(
            f"Used (additional) memory: {mem_used_MB_after - mem_used_MB_before:.2f} MB"
        )

    def prepare(self):
        patch_vertex_coords = self.patch_vertex_coords
        center = patch_vertex_coords.mean(-2)
        valid_radiance_ids = self.valid_radiance_ids

        radiance_pos = center[valid_radiance_ids[0]]
        radiance_dir = center[valid_radiance_ids[1]] - radiance_pos
        radiance_dir = F.normalize(radiance_dir, dim=-1)

        pos_min = radiance_pos.min(0, keepdim=True)[0]
        pos_max = radiance_pos.max(0, keepdim=True)[0]
        radiance_pos_normalized = 2 * (radiance_pos - pos_min) / (pos_max - pos_min) - 1

        self.register_buffer("radiance_pos", radiance_pos)
        self.register_buffer("radiance_pos_normalized", radiance_pos_normalized)
        self.register_buffer("radiance_dir", radiance_dir)
        self.register_buffer("pos_min", pos_min)
        self.register_buffer("pos_max", pos_max)

        self.pos_normalize = lambda x: (
            2 * (x - self.pos_min) / (self.pos_max - self.pos_min) - 1
        )

    # @profile
    def forward(
        self,
        source_pos,
        receiver_pos,
        absorption_coefficient,
        scattering_coefficient,
        delta_kernel=None,
        return_all_intermediates=False,
    ):
        scattering_matrix = self.compose_kernel(
            absorption_coefficient, scattering_coefficient, delta_kernel=delta_kernel
        )
        initial_radiance, delays = self.injection(
            source_pos,
            scattering_matrix=scattering_matrix,
            direction=self.injection_direction,
            return_minimum_delays=True,
        )
        radiance, intermediates = self.compute_radiance(
            initial_radiance,
            scattering_matrix,
            return_all_intermediates=return_all_intermediates,
        )
        echogram = self.detection(
            receiver_pos, radiance=radiance, direction=self.detection_direction
        )
        if self.direct_arrival:
            direct_echogram = compute_direct_component(
                source_pos=source_pos,
                receiver_pos=receiver_pos,
                patch_vertex_coords=self.patch_vertex_coords,
                radiance_sampling_rate=self.radiance_sampling_rate,
                speed_of_sound=self.speed_of_sound,
                echogram_len=self.echogram_len,
                source_directivity=None,
                source_orentation=None,
                receiver_directivity=None,
                receiver_orientation=None,
            )
            echogram = echogram + direct_echogram
        else:
            direct_echogram = None

        if self.learnable_envelope:
            envelope = torch.exp(self.envelope)
            echogram = echogram * envelope

        return (
            echogram,
            initial_radiance,
            radiance,
            delays,
            direct_echogram,
            intermediates,
        )

    def compose_kernel(
        self, absorption_coefficient, scattering_coefficient, delta_kernel=None
    ):
        absorption_coefficient = absorption_coefficient[self.sparse_kernel_reflector_id]
        scattering_coefficient = scattering_coefficient[
            :, self.sparse_kernel_reflector_id
        ]
        lossless_matrix = (scattering_coefficient * self.sparse_kernel_basis).sum(0)
        scattering_matrix = lossless_matrix * absorption_coefficient
        if delta_kernel is not None:
            scattering_matrix = F.relu(scattering_matrix + delta_kernel)
        scattering_matrix = SparseTensor(
            row=self.sparse_kernel_row,
            col=self.sparse_kernel_col,
            value=scattering_matrix,
            sparse_sizes=(self.num_radiances, self.num_radiances),
        )
        return scattering_matrix

    def compute_radiance(
        self, initial_radiance, scattering_matrix, return_all_intermediates=False
    ):
        match self.main_loop_domain:
            case "frequency":
                if self.fsm_correection:
                    initial_radiance = initial_radiance * self.fsm_window
                    delay_signal = self.delay_signal * self.fsm_window
                radiance, intermediates = main_art_loop_sparse_mm(
                    initial_radiance,
                    delay_signal,
                    scattering_matrix,
                    num_bounces=self.num_bounces,
                    return_all_intermediates=return_all_intermediates,
                )
                if self.fsm_correection:
                    radiance = radiance / self.fsm_window
                    if return_all_intermediates:
                        intermediates = [
                            intermediate / self.fsm_window
                            for intermediate in intermediates
                        ]
            case "time":
                radiance = main_art_loop_sparse_mm_exact_delay(
                    initial_radiance,
                    self.delay_samples,
                    scattering_matrix,
                    num_bounces=self.num_bounces,
                )
        return radiance, intermediates

