"""
Differentiable Acoustic Radiance Transfer (DART), a patch-direction (PD) variant (discussed in Appendix).
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
from torchaudio.functional import fftconvolve

from torch_geometric_acoustics.art.direct import compute_direct_component
from torch_geometric_acoustics.art.directivity import LearnableDirectivity
from torch_geometric_acoustics.art.geometry import integrated_geometry_patch_direction
from torch_geometric_acoustics.art.injection_detection.patch_direction import (
    compute_inject_radiance,
    detect_echogram,
)
from torch_geometric_acoustics.art.kernel import compute_reflection_kernel_basis
from torch_geometric_acoustics.art.kernel.patch_direction_factorized import (
    compose_block_diag,
    compute_local_kernel,
)
from torch_geometric_acoustics.art.kernel.sparse import (
    compile_basis_kernels,
    postprocess_basis_kernels,
)
from torch_geometric_acoustics.art.main_loop import (
    main_art_loop_sparse_mm,
    main_art_loop_sparse_mm_exact_delay,
)
from torch_geometric_acoustics.core import (
    compute_bin_centers,
    compute_local_orthonomal_matrix,
    compute_patch_geometric_properties,
    delay_impulse,
    sample_direction,
)


class AcousticRadianceTransfer_PatchDirection(nn.Module):
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
        N_ele=16,
        N_azi=16,
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
        self.N_ele = N_ele
        self.N_azi = N_azi

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

    @torch.no_grad()
    def precompute(self):
        print("Precomputing...")

        free, total = torch.cuda.mem_get_info()
        mem_used_MB_before = (total - free) / 1024**2

        # --------------------------------
        # geometry & kernel
        geometry = integrated_geometry_patch_direction(
            area=self.area, N_azi=self.N_azi, N_ele=self.N_ele
        )
        local_orthonomal_matrix = compute_local_orthonomal_matrix(
            self.patch_vertex_coords
        )

        kernel_basis, average_distance = compute_reflection_kernel_basis(
            method="patch-direction",
            patch_vertex_coords=self.patch_vertex_coords,
            normal=self.normal,
            brdfs=self.brdfs,
            N_ele=self.N_ele,
            N_azi=self.N_azi,
            dense_output=False,
        )
        self.register_buffer("local_orthonomal_matrix", local_orthonomal_matrix)

        local_kernel = compute_local_kernel(
            brdfs=self.brdfs,
            N_azi=self.N_azi,
            N_ele=self.N_ele,
            num_patches=self.num_patches,
        )
        injection_kernel_basis = torch.stack(
            [local_kernel[brdf] for brdf in self.brdfs]
        )
        self.register_buffer("injection_kernel_basis", injection_kernel_basis)
        print("injection_kernel_basis", injection_kernel_basis.shape)

        # --------------------------------
        # sparse
        radiance_mask = geometry > 0  # SURFACE BOTH SIDES ??? HANDLE HERE
        kernel_basis, nonzero_radiance_mask = postprocess_basis_kernels(
            kernels=kernel_basis,
            radiance_mask=radiance_mask,
        )
        print(kernel_basis)

        geometry = geometry[nonzero_radiance_mask]
        average_distance = average_distance[nonzero_radiance_mask]

        valid_radiance_ids = list(
            itertools.product(range(self.num_patches), range(self.N_azi * self.N_ele))
        )
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
        self.register_buffer("nonzero_radiance_mask", nonzero_radiance_mask)

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
            compute_inject_radiance,
            patch_vertex_coords=self.patch_vertex_coords,
            valid_radiance_ids=self.valid_radiance_ids,
            geometry=self.geometry,
            local_orthonomal_matrix=self.local_orthonomal_matrix,
            num_rays=self.num_injection_rays,
            echogram_len=self.echogram_len,
            radiance_sampling_rate=self.radiance_sampling_rate,
            speed_of_sound=self.speed_of_sound,
            sampling_method=self.direction_sampling_method,
            aggregate_delay=True,
            N_ele=self.N_ele,
            N_azi=self.N_azi,
        )

        self.detection = partial(
            detect_echogram,
            patch_vertex_coords=self.patch_vertex_coords,
            valid_radiance_ids=self.valid_radiance_ids,
            num_rays=self.num_detection_rays,
            echogram_len=self.echogram_len,
            local_orthonomal_matrix=self.local_orthonomal_matrix,
            radiance_sampling_rate=self.radiance_sampling_rate,
            speed_of_sound=self.speed_of_sound,
            sampling_method=self.direction_sampling_method,
            geometry=self.geometry,
            N_ele=self.N_ele,
            N_azi=self.N_azi,
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
        device = patch_vertex_coords.device

        radiance_pos = center[valid_radiance_ids[0]]
        radiance_dir = compute_bin_centers(
            N_ele=self.N_ele, N_azi=self.N_azi, device=device
        )
        radiance_dir = radiance_dir.view(-1, 3)
        radiance_dir = torch.einsum(
            "mij,nj->mni", self.local_orthonomal_matrix, radiance_dir
        )
        radiance_dir = radiance_dir[valid_radiance_ids[0], valid_radiance_ids[1]]

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

    def forward(
        self,
        source_pos,
        receiver_pos,
        absorption_coefficient,
        scattering_coefficient,
        source_orientation=None,
        source_directivity=None,
        delta_kernel=None,
        return_all_intermediates=False,
        delay_signal=None,
        envelope=None,
        radiance_gain=None,
        direct_gain=None,
        post_conv=None,
    ):
        injection_scttering_matrix, scattering_matrix = self.compose_kernel(
            absorption_coefficient, scattering_coefficient, delta_kernel=delta_kernel
        )
        initial_radiance, delays = self.injection(
            source_pos,
            source_orientation=source_orientation,
            source_directivity=source_directivity,
            direction=self.injection_direction,
            injection_scattering_matrix=injection_scttering_matrix,
            return_minimum_delays=True,
        )
        radiance, intermediates = self.compute_radiance(
            initial_radiance,
            scattering_matrix,
            delay_signal=delay_signal,
            return_all_intermediates=return_all_intermediates,
        )
        echogram = self.detection(
            receiver_pos, radiance=radiance, direction=self.detection_direction
        )
        if radiance_gain is not None:
            echogram = echogram * radiance_gain

        if self.direct_arrival:
            direct_echogram = compute_direct_component(
                source_pos=source_pos,
                receiver_pos=receiver_pos,
                patch_vertex_coords=self.patch_vertex_coords,
                radiance_sampling_rate=self.radiance_sampling_rate,
                speed_of_sound=self.speed_of_sound,
                echogram_len=self.echogram_len,
                source_directivity=source_directivity,
                source_orientation=source_orientation,
                receiver_directivity=None,
                receiver_orientation=None,
            )
            if direct_gain is not None:
                direct_echogram = direct_echogram * direct_gain
            echogram = echogram + direct_echogram
        else:
            direct_echogram = None

        if envelope is not None:
            echogram = echogram * envelope

        if post_conv is not None:
            echogram = fftconvolve(echogram, post_conv, mode="full")
            half = len(post_conv) // 2
            slice_from, slice_to = half, -half
            echogram = echogram[slice_from:slice_to]

        echogram = torch.clamp(echogram, min=0.0)

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
        absorption_coefficient_sparse = absorption_coefficient[
            self.sparse_kernel_reflector_id
        ]
        scattering_coefficient_sparse = scattering_coefficient[
            :, self.sparse_kernel_reflector_id
        ]

        lossless_matrix = (
            scattering_coefficient_sparse * self.sparse_kernel_basis
        ).sum(0)
        scattering_matrix = lossless_matrix * absorption_coefficient_sparse
        if delta_kernel is not None:
            scattering_matrix = F.relu(scattering_matrix + delta_kernel)
        scattering_matrix = SparseTensor(
            row=self.sparse_kernel_row,
            col=self.sparse_kernel_col,
            value=scattering_matrix,
            sparse_sizes=(self.num_radiances, self.num_radiances),
        )

        injection_scattering_matrix = (
            scattering_coefficient[:, :, None, None] * self.injection_kernel_basis
        ).sum(0)
        injection_scattering_matrix = (
            injection_scattering_matrix * absorption_coefficient[:, None, None]
        )
        injection_scattering_matrix = compose_block_diag(injection_scattering_matrix)
        injection_scattering_matrix = injection_scattering_matrix[
            self.nonzero_radiance_mask, :
        ]
        injection_scattering_matrix = injection_scattering_matrix[
            :, self.nonzero_radiance_mask
        ]

        return injection_scattering_matrix, scattering_matrix

    def compute_radiance(
        self,
        initial_radiance,
        scattering_matrix,
        delay_signal=None,
        return_all_intermediates=False,
    ):
        if delay_signal is None:
            delay_signal = self.delay_signal
        match self.main_loop_domain:
            case "frequency":
                if self.fsm_correection:
                    initial_radiance = initial_radiance * self.fsm_window
                    delay_signal = delay_signal * self.fsm_window
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


class PDART_Learnable(AcousticRadianceTransfer_PatchDirection):
    def __init__(
        self,
        learnable_feedback_delay=False,
        learnable_air_absorption=False,
        learnable_envelope=False,
        learnable_direct_gain=True,
        learnable_postconv=False,
        directional_source=False,
        **art_kwargs,
    ):
        super().__init__(**art_kwargs)

        self.learnable_feedback_delay = learnable_feedback_delay
        self.learnable_air_absorption = learnable_air_absorption
        self.learnable_envelope = learnable_envelope
        self.learnable_direct_gain = learnable_direct_gain
        self.learnable_postconv = learnable_postconv
        self.directional_source = directional_source

        absorption_coefficient_z = torch.zeros(self.num_patches)
        scattering_coefficient_z = torch.zeros(self.num_brdfs, self.num_patches)
        self.absorption_coefficient_z = nn.Parameter(absorption_coefficient_z)
        self.scattering_coefficient_z = nn.Parameter(scattering_coefficient_z)

        if self.learnable_air_absorption:
            self.air_absorption_z = nn.Parameter(torch.zeros(1))

        if self.learnable_envelope:
            self.envelope_z = nn.Parameter(torch.zeros(self.echogram_len))

        if self.learnable_direct_gain:
            self.direct_gain_z = nn.Parameter(torch.zeros(1))
            self.radiance_gain_z = nn.Parameter(torch.zeros(1))
        else:
            self.gain_z = nn.Parameter(torch.zeros(1))

        if self.learnable_postconv:
            postconv_len = 0.025 * self.radiance_sampling_rate
            postconv_len = 2 * int(postconv_len) + 1
            postconv_len = int(postconv_len)

            postconv = torch.zeros(postconv_len)
            postconv[postconv_len // 2] = 1
            self.postconv_z = nn.Parameter(postconv)

        if self.directional_source:
            self.directivity = LearnableDirectivity()
        else:
            self.directivity = None

    def precompute(self):
        super().precompute()

        if self.learnable_feedback_delay:
            self.feedback_delay_z = nn.Parameter(torch.zeros(self.num_radiances))

    def forward(self, source_pos, receiver_pos, source_orientation=None):
        absorption_coefficient = torch.sigmoid(self.absorption_coefficient_z)
        scattering_coefficient = torch.softmax(self.scattering_coefficient_z, 0)

        if self.learnable_envelope:
            envelope = F.softplus(self.envelope_z)
        else:
            envelope = None

        if self.learnable_feedback_delay:
            delay_offset = torch.tanh(self.feedback_delay_z) * 5
            delays = self.delay_samples + delay_offset
            delays = delays.clamp(min=0)
            delay_signal = delay_impulse(
                delay_samples=delays,
                signal_len=self.echogram_len,
                method="fraction_linear",
            )
            if self.learnable_air_absorption:
                distance_offset = delay_offset * (
                    self.speed_of_sound / self.radiance_sampling_rate
                )
                distance = self.average_distance + distance_offset
                delay_gain_db = F.softplus(self.air_absorption_z) / 100
                delay_gain = 10 ** (-delay_gain_db * distance)
                delay_signal = delay_signal * delay_gain[:, None]
        else:
            delay_signal = None

        if self.learnable_direct_gain:
            radiance_gain = F.softplus(self.radiance_gain_z)
            direct_gain = F.softplus(self.direct_gain_z)
        else:
            gain = F.softplus(self.gain_z)
            radiance_gain = gain
            direct_gain = gain

        if self.learnable_postconv:
            post_conv = F.softplus(self.postconv_z)
            post_conv = post_conv / post_conv.sum()
        else:
            post_conv = None

        pred_echogram, _, _, _, _, _ = super().forward(
            source_pos=source_pos,
            source_orientation=source_orientation,
            source_directivity=self.directivity,
            receiver_pos=receiver_pos,
            absorption_coefficient=absorption_coefficient,
            scattering_coefficient=scattering_coefficient,
            envelope=envelope,
            delay_signal=delay_signal,
            radiance_gain=radiance_gain,
            direct_gain=direct_gain,
            post_conv=post_conv,
        )
        return absorption_coefficient, None, pred_echogram
