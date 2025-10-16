"""
Differentiable Acoustic Radiance Transfer (DART), a patch-direction-factorized (PDF) variant.
This version of DART is used for all the experiments in the main paper.
Currently supports nonparametric and parametric variant with four BSDFs:
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
from torch_geometric_acoustics.art.direct import compute_direct_component
from torch_geometric_acoustics.art.directivity import (
    LearnableAxialDirectivity,
)
from torch_geometric_acoustics.art.geometry import (
    integrated_geometry_patch_direction,
)
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
from torch_geometric_acoustics.image_source_method import single_order_ism
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torchaudio.functional import fftconvolve


class AcousticRadianceTransfer_PatchDirectionFactorized(nn.Module):
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
        inject_ism=False,
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
        object_ids = torch.tensor(list(mesh.object_ids.values()), dtype=torch.long)
        _, object_ids = torch.unique(object_ids, return_inverse=True)

        self.register_buffer("patch_vertex_coords", patch_vertex_coords)
        self.register_buffer("normal", normal)
        self.register_buffer("area", area)
        self.register_buffer("object_ids", object_ids)

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
        self.inject_ism = inject_ism

        if inject_ism:
            assert len(brdfs) > 0 and brdfs[0] == "specular"

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

        self.local_kernel = nn.Parameter()

    @torch.no_grad()
    def precompute(self):
        print("Precomputing...")

        free, total = torch.cuda.mem_get_info()
        mem_used_MB_before = (total - free) / 1024**2

        # --------------------------------
        # orthonomal matrix
        local_orthonomal_matrix = compute_local_orthonomal_matrix(
            self.patch_vertex_coords
        )
        self.register_buffer("local_orthonomal_matrix", local_orthonomal_matrix)

        # --------------------------------
        # geometry
        geometry = integrated_geometry_patch_direction(
            area=self.area, N_azi=self.N_azi, N_ele=self.N_ele
        )

        # --------------------------------
        # kernel (global)
        kernel_basis, average_distance = compute_reflection_kernel_basis(
            method="patch-direction-factorized",
            patch_vertex_coords=self.patch_vertex_coords,
            normal=self.normal,
            brdfs=[],
            N_ele=self.N_ele,
            N_azi=self.N_azi,
            # dense_output=False,
        )
        radiance_mask = geometry > 0  
        kernel_basis, nonzero_radiance_mask = postprocess_basis_kernels(
            kernels=kernel_basis,
            radiance_mask=radiance_mask,
        )
        global_kernel = kernel_basis["global"]
        row, col, val = global_kernel.coo()
        size = global_kernel.sparse_sizes()

        self.register_buffer("global_kernel_row", row)
        self.register_buffer("global_kernel_col", col)
        self.register_buffer("global_kernel_val", val)
        self.global_kernel_size = size

        if len(self.brdfs) > 0:
            local_kernel_basis = compute_local_kernel(
                brdfs=self.brdfs,
                N_azi=self.N_azi,
                N_ele=self.N_ele,
                num_patches=self.num_patches,
            )
            local_kernel_basis = torch.stack(
                [local_kernel_basis[brdf] for brdf in self.brdfs]
            )
            self.register_buffer("local_kernel_basis", local_kernel_basis)
            print("local_kernel_basis", local_kernel_basis.shape)

        # --------------------------------
        # sparse
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

        self.register_buffer("geometry", geometry)
        self.register_buffer("valid_radiance_ids", valid_radiance_ids)
        self.register_buffer("nonzero_radiance_mask", nonzero_radiance_mask)

        # ---------------------------------
        # delay
        delay_samples = average_distance * (
            self.radiance_sampling_rate / self.speed_of_sound
        )
        delay_signal = delay_impulse(
            delay_samples=delay_samples,
            signal_len=self.echogram_len,
            method="fraction_linear",
        )
        delay_samples_int = delay_samples.round().long()
        self.register_buffer("average_distance", average_distance)
        self.register_buffer("delay_signal", delay_signal)
        self.register_buffer("delay_samples", delay_samples)
        self.register_buffer("delay_samples_int", delay_samples_int)

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
        print("Some stats:")
        print(f"  - Number of patches: {self.num_patches}")
        print(f"  - Number of radiances: {self.num_radiances}")

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
        scattering_coefficient=None,
        source_orientation=None,
        source_directivity=None,
        local_kernel=None,
        delta_kernel=None,
        delay_signal=None,
        envelope=None,
        radiance_gain=None,
        direct_gain=None,
        ism_gain=None,
        return_all_intermediates=False,
        post_conv=None,
    ):
        (
            local_kernel,
            local_specular_kernel,
            local_residual_kernel,
            local_kernel_dense,
        ) = self.compose_local_kernel(
            absorption_coefficient,
            scattering_coefficient,
            local_kernel=local_kernel,
            delta_kernel=delta_kernel,
        )
        global_kernel = SparseTensor(
            row=self.global_kernel_row,
            col=self.global_kernel_col,
            value=self.global_kernel_val,
            sparse_sizes=self.global_kernel_size,
        )
        initial_radiance, initial_residual_radiance, delays = self.injection(
            source_pos=source_pos,
            source_orientation=source_orientation,
            source_directivity=source_directivity,
            direction=self.injection_direction,
            injection_scattering_matrix=local_kernel,
            injection_residual_matrix=local_residual_kernel,
            return_minimum_delays=True,
        )
        radiance, intermediates = self.compute_radiance(
            initial_radiance,
            [global_kernel, local_kernel],
            delay_signal=delay_signal,
            return_all_intermediates=return_all_intermediates,
            bounce_after=1 if self.inject_ism else 0,
        )
        if self.inject_ism:
            radiance = radiance + initial_residual_radiance
        echogram = self.detection(
            receiver_pos, radiance=radiance, direction=self.detection_direction
        )
        if radiance_gain is not None:
            echogram = echogram * radiance_gain

        if self.direct_arrival:
            direct_echogram = compute_direct_component(
                source_pos=source_pos,
                source_orientation=source_orientation,
                source_directivity=source_directivity,
                receiver_pos=receiver_pos,
                patch_vertex_coords=self.patch_vertex_coords,
                radiance_sampling_rate=self.radiance_sampling_rate,
                speed_of_sound=self.speed_of_sound,
                echogram_len=self.echogram_len,
                receiver_directivity=None,
                receiver_orientation=None,
            )
            if direct_gain is not None:
                direct_echogram = direct_echogram * direct_gain
            echogram = echogram + direct_echogram
        else:
            direct_echogram = None

        if self.inject_ism:
            ism_echogram = single_order_ism(
                patch_vertex_coords=self.patch_vertex_coords,
                normal=self.normal,
                source_pos=source_pos,
                receiver_pos=receiver_pos,
                radiance_sampling_rate=self.radiance_sampling_rate,
                echogram_len=self.echogram_len,
                speed_of_sound=self.speed_of_sound,
                source_directivity=source_directivity,
                source_orientation=source_orientation,
                absorption_coefficient=absorption_coefficient,
                scattering_coefficient=scattering_coefficient,
            )
            if ism_gain is not None:
                ism_echogram = ism_echogram * ism_gain
            echogram = echogram + ism_echogram

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
            local_kernel_dense,
        )

    def compose_local_kernel(
        self,
        absorption_coefficient,
        scattering_coefficient=None,
        local_kernel=None,
        delta_kernel=None,
    ):
        def compress(x):
            x = compose_block_diag(x)
            x = x[self.nonzero_radiance_mask, :]
            x = x[:, self.nonzero_radiance_mask]
            return x

        if scattering_coefficient is not None:
            scattering_coefficient = scattering_coefficient[:, :, None, None]
        if self.inject_ism:
            if local_kernel is None:
                local_kernel = scattering_coefficient * self.local_kernel_basis
                local_kernel = (
                    local_kernel * absorption_coefficient[None, :, None, None]
                )
                local_specular_kernel = local_kernel[0]
                local_residual_kernel = local_kernel[1:].sum(0)
            else:
                local_kernel = torch.stack(
                    [self.local_kernel_basis[0], local_kernel], 0
                )
                local_kernel = scattering_coefficient * local_kernel
                local_kernel = (
                    local_kernel * absorption_coefficient[None, :, None, None]
                )
                local_specular_kernel = local_kernel[0]
                local_residual_kernel = local_kernel[1:].sum(0)

            local_kernel_dense = local_specular_kernel + local_residual_kernel
            local_specular_kernel = compress(local_specular_kernel)
            local_residual_kernel = compress(local_residual_kernel)
            local_kernel = local_specular_kernel + local_residual_kernel

        else:
            if local_kernel is None:
                local_kernel = (scattering_coefficient * self.local_kernel_basis).sum(0)
            local_kernel = local_kernel * absorption_coefficient[:, None, None]
            local_kernel_dense = local_kernel
            local_kernel = compress(local_kernel)
            local_specular_kernel = None
            local_residual_kernel = None
        return (
            local_kernel,
            local_specular_kernel,
            local_residual_kernel,
            local_kernel_dense,
        )

    # @profile
    def compute_radiance(
        self,
        initial_radiance,
        scattering_matrix,
        delay_signal=None,
        return_all_intermediates=False,
        bounce_after=1,
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
                    bounce_after=bounce_after,
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
                    bounce_after=bounce_after,
                )
        return radiance, intermediates


class PDFART_Learnable(AcousticRadianceTransfer_PatchDirectionFactorized):
    def __init__(
        self,
        learnable_feedback_delay=False,
        learnable_air_absorption=False,
        learnable_envelope=True,
        learnable_direct_gain=True,
        learnable_postconv=False,
        directional_source=False,
        kernel_regularization=True,
        inject_ism=False,
        shared_param=False,
        reflection_only=False,
        **art_kwargs,
    ):
        self.inject_ism = inject_ism
        if inject_ism:
            brdfs = ["specular"]
        else:
            brdfs = []
        super().__init__(**art_kwargs, inject_ism=inject_ism, brdfs=brdfs)

        self.learnable_feedback_delay = learnable_feedback_delay
        self.learnable_air_absorption = learnable_air_absorption
        self.learnable_envelope = learnable_envelope
        self.learnable_direct_gain = learnable_direct_gain
        self.learnable_postconv = learnable_postconv
        self.directional_source = directional_source
        self.kernel_regularization = kernel_regularization
        self.shared_param = shared_param
        self.num_objects = 1 + torch.max(self.object_ids)
        self.reflection_only = reflection_only

        if self.shared_param:
            N = self.num_objects
        else:
            N = self.num_patches

        absorption_coefficient_z = torch.zeros(N)
        self.absorption_coefficient_z = nn.Parameter(absorption_coefficient_z)
        if self.inject_ism:
            scattering_coefficient_z = torch.zeros(2, N)
            self.scattering_coefficient_z = nn.Parameter(scattering_coefficient_z)

        num_directions = self.N_azi * self.N_ele
        local_kernel_z = torch.zeros(N, num_directions, num_directions)
        self.local_kernel_z = nn.Parameter(local_kernel_z)

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
            self.directivity = LearnableAxialDirectivity()
        else:
            self.directivity = None

        if self.reflection_only:
            block = torch.ones(
                num_directions // 2, num_directions // 2, dtype=torch.bool
            )
            reflection_only_mask = torch.block_diag(block, block)
            self.register_buffer("reflection_only_mask", reflection_only_mask)
        if self.inject_ism:
            self.ism_gain_z = nn.Parameter(torch.zeros(1))

    def precompute(self):
        super().precompute()

        if self.learnable_feedback_delay:
            self.feedback_delay_z = nn.Parameter(torch.zeros(self.num_radiances))

    def forward(self, source_pos, receiver_pos, source_orientation=None):
        absorption_coefficient = torch.sigmoid(self.absorption_coefficient_z)

        local_kernel_z = self.local_kernel_z
        if self.reflection_only:
            local_kernel_z_ = torch.zeros_like(local_kernel_z)
            local_kernel_z_[:, self.reflection_only_mask] = local_kernel_z[
                :, self.reflection_only_mask
            ]
            local_kernel_z_[:, ~self.reflection_only_mask] = -100
            local_kernel = local_kernel_z_.softmax(-1)
        else:
            local_kernel = local_kernel_z.softmax(-1)

        if self.shared_param:
            absorption_coefficient = absorption_coefficient[self.object_ids]
            local_kernel = local_kernel[self.object_ids]

        if self.inject_ism:
            scattering_coefficient = torch.softmax(self.scattering_coefficient_z, 0)
            if self.shared_param:
                scattering_coefficient = scattering_coefficient[:, self.object_ids]
        else:
            scattering_coefficient = None

        if self.learnable_envelope:
            envelope = torch.exp(self.envelope_z)
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
            radiance_gain = torch.exp(self.radiance_gain_z)
            direct_gain = torch.exp(self.direct_gain_z)
        else:
            gain = F.softplus(self.gain_z)
            radiance_gain = gain
            direct_gain = gain

        if self.inject_ism:
            ism_gain = torch.exp(self.ism_gain_z)
        else:
            ism_gain = None

        if self.learnable_postconv:
            post_conv = F.softplus(self.postconv_z)
            post_conv = post_conv / post_conv.sum()
        else:
            post_conv = None

        pred_echogram, _, _, _, _, _, _ = super().forward(
            source_pos=source_pos,
            source_orientation=source_orientation,
            source_directivity=self.directivity,
            receiver_pos=receiver_pos,
            absorption_coefficient=absorption_coefficient,
            scattering_coefficient=scattering_coefficient,
            local_kernel=local_kernel,
            envelope=envelope,
            delay_signal=delay_signal,
            radiance_gain=radiance_gain,
            direct_gain=direct_gain,
            ism_gain=ism_gain,
            post_conv=post_conv,
        )
        if self.learnable_envelope:
            regularization_loss_dict = {
                "envelope_reg_loss": self.envelope_z.abs().mean()
            }
        else:
            regularization_loss_dict = {"envelope_reg_loss": 0.0}
        if self.kernel_regularization:
            kernel_regularization_loss_dict = self.compute_regularization_loss(
                local_kernel
            )
            regularization_loss_dict = {
                **regularization_loss_dict,
                **kernel_regularization_loss_dict,
            }
            return absorption_coefficient, None, pred_echogram, regularization_loss_dict
        else:
            return absorption_coefficient, None, pred_echogram

    # @profile
    def compute_regularization_loss(
        self,
        local_kernel,
    ):
        if self.shared_param:
            obj_group_loss = 0.0
        else:
            ele = np.random.randint(self.N_ele)
            azi = np.random.randint(self.N_azi)
            single_out = local_kernel[:, ele * self.N_azi + azi, :]
            mean_stat = scatter(single_out, self.object_ids, dim=0, reduce="mean")
            obj_group_loss = (single_out - mean_stat[self.object_ids]).abs().mean()

        ele = np.random.randint(self.N_ele)
        local_kernel = local_kernel.view(
            -1, self.N_ele, self.N_azi, self.N_ele, self.N_azi
        )
        shift = np.random.randint(self.N_azi)
        shifted_kernel = local_kernel.roll(shift, 2).roll(shift, 4)

        rotation_symmetric_loss = (local_kernel - shifted_kernel).abs().mean()

        azi = np.random.randint(self.N_azi)
        kernel_azi = local_kernel[:, :, :, :, azi]
        device = local_kernel.device
        idxs = torch.arange(self.N_azi, device=device)
        left = (azi - idxs) % self.N_azi
        right = (idxs - azi) % self.N_azi
        reflection_symmetric_loss = (
            (kernel_azi[:, :, left] - kernel_azi[:, :, right]).abs().mean()
        )

        kernel_reg_loss = (
            obj_group_loss + rotation_symmetric_loss + reflection_symmetric_loss
        )
        return dict(
            obj_group_loss=obj_group_loss,
            rotation_symmetric_loss=rotation_symmetric_loss,
            reflection_symmetric_loss=reflection_symmetric_loss,
            kernel_reg_loss=kernel_reg_loss,
        )


class PDFART_Learnable_Parametric(AcousticRadianceTransfer_PatchDirectionFactorized):
    def __init__(
        self,
        learnable_feedback_delay=False,
        learnable_air_absorption=False,
        learnable_envelope=True,
        learnable_direct_gain=True,
        learnable_postconv=False,
        directional_source=False,
        kernel_regularization=True,
        shared_param=False,
        **art_kwargs,
    ):
        super().__init__(**art_kwargs)

        self.learnable_feedback_delay = learnable_feedback_delay
        self.learnable_air_absorption = learnable_air_absorption
        self.learnable_envelope = learnable_envelope
        self.learnable_direct_gain = learnable_direct_gain
        self.learnable_postconv = learnable_postconv
        self.directional_source = directional_source
        self.kernel_regularization = kernel_regularization
        self.shared_param = shared_param
        self.num_objects = 1 + torch.max(self.object_ids)

        if self.shared_param:
            N = self.num_objects
        else:
            N = self.num_patches

        absorption_coefficient_z = torch.zeros(N)
        scattering_coefficient_z = torch.zeros(self.num_brdfs, N)
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
            self.directivity = LearnableAxialDirectivity()
        else:
            self.directivity = None

    def precompute(self):
        super().precompute()

        if self.learnable_feedback_delay:
            self.feedback_delay_z = nn.Parameter(torch.zeros(self.num_radiances))

    def forward(self, source_pos, receiver_pos, source_orientation=None):
        absorption_coefficient = torch.sigmoid(self.absorption_coefficient_z)
        scattering_coefficient = torch.softmax(self.scattering_coefficient_z, 0)

        if self.shared_param:
            absorption_coefficient = absorption_coefficient[self.object_ids]
            scattering_coefficient = scattering_coefficient[:, self.object_ids]

        if self.learnable_envelope:
            envelope = torch.exp(self.envelope_z)
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
            radiance_gain = torch.exp(self.radiance_gain_z)
            direct_gain = torch.exp(self.direct_gain_z)
        else:
            gain = F.softplus(self.gain_z)
            radiance_gain = gain
            direct_gain = gain

        if self.learnable_postconv:
            post_conv = F.softplus(self.postconv_z)
            post_conv = post_conv / post_conv.sum()
        else:
            post_conv = None

        pred_echogram, _, _, _, _, _, local_kernel = super().forward(
            source_pos=source_pos,
            source_orientation=source_orientation,
            source_directivity=self.directivity,
            receiver_pos=receiver_pos,
            absorption_coefficient=absorption_coefficient,
            scattering_coefficient=scattering_coefficient,
            local_kernel=None,
            envelope=envelope,
            delay_signal=delay_signal,
            radiance_gain=radiance_gain,
            direct_gain=direct_gain,
            post_conv=post_conv,
        )

        if self.learnable_envelope:
            regularization_loss_dict = {
                "envelope_reg_loss": self.envelope_z.abs().mean()
            }
        else:
            regularization_loss_dict = {"envelope_reg_loss": 0.0}
        if self.kernel_regularization:
            kernel_regularization_loss_dict = self.compute_regularization_loss(
                local_kernel
            )
            regularization_loss_dict = {
                **regularization_loss_dict,
                **kernel_regularization_loss_dict,
            }
            return absorption_coefficient, None, pred_echogram, regularization_loss_dict
        else:
            return absorption_coefficient, None, pred_echogram

    def compute_regularization_loss(
        self,
        local_kernel,
    ):
        if self.shared_param:
            return dict(
                obj_group_loss=0.0,
                kernel_reg_loss=0.0,
            )
        else:
            ele = np.random.randint(self.N_ele)
            azi = np.random.randint(self.N_azi)
            single_out = local_kernel[:, ele * self.N_azi + azi, :]
            mean_stat = scatter(single_out, self.object_ids, dim=0, reduce="mean")
            obj_group_loss = (single_out - mean_stat[self.object_ids]).abs().mean()

            kernel_reg_loss = obj_group_loss
            return dict(
                obj_group_loss=obj_group_loss,
                kernel_reg_loss=kernel_reg_loss,
            )
