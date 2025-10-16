r"""
The main ART operation that reflects and scatters the radiance.
All done in frequency domain with sparse matrix multiplications.
Note that the antialiasing is done in inside the ART module.
"""

import torch
import torch.fft
from torch_geometric_acoustics.core import apply_integer_delay
from torch_sparse import SparseTensor


def main_art_loop_sparse_mm(
    radiance,
    delay,
    reflection_kernel,
    num_bounces=50,
    bounce_after=0,
    return_all_intermediates=False,
):
    r"""
    Main loop for the differentiable ART model

    radiance: [N, T], float tensor
    delay: [N, T], float tensor
    reflection_kernel: [N, N], float tensor (sparse)
    """
    n, t = radiance.shape
    f = t // 2 + 1
    assert radiance.ndim == 2 & delay.ndim == 2

    radiance = torch.fft.rfft(radiance)
    delay = torch.fft.rfft(delay)
    if bounce_after == 0:
        transfered_radiance = radiance
    else:
        transfered_radiance = 0.0
    intermediates = []
    for i in range(num_bounces):
        radiance = radiance * delay
        radiance = torch.view_as_real(radiance)
        radiance = radiance.view(n, f * 2)

        if isinstance(reflection_kernel, SparseTensor):
            radiance = reflection_kernel.matmul(radiance)
        elif isinstance(reflection_kernel, list):
            for kernel in reflection_kernel:
                radiance = kernel.matmul(radiance)

        radiance = radiance.view(n, f, 2)
        radiance = torch.view_as_complex(radiance)
        if return_all_intermediates:
            radiance_t = torch.fft.irfft(radiance)
            intermediates.append(radiance_t)
        if i >= bounce_after:
            transfered_radiance = transfered_radiance + radiance
    transfered_radiance = torch.fft.irfft(transfered_radiance)
    if return_all_intermediates:
        return transfered_radiance, intermediates
    else:
        return transfered_radiance, None


def main_art_loop_sparse_mm_exact_delay(
    radiance, delay, reflection_kernel, num_bounces=50
):
    r"""
    Main loop for the differentiable ART model

    radiance: [N, T], float tensor
    delay: [N, T], float tensor
    reflection_kernel: [N, N], float tensor (sparse)
    """
    n, t = radiance.shape
    transfered_radiance = radiance
    for _ in range(num_bounces):
        radiance = apply_integer_delay(radiance, delay)
        radiance = reflection_kernel.matmul(radiance)
        transfered_radiance = transfered_radiance + radiance
    return transfered_radiance


def single_bounce_exact_delay(radiance, delay, reflection_kernel):
    radiance = apply_integer_delay(radiance, delay)
    radiance = reflection_kernel.matmul(radiance)
    return radiance


def single_bounce(radiance, delay_signal, reflection_kernel):
    n, t = radiance.shape
    f = t // 2 + 1
    assert radiance.ndim == 2 & delay_signal.ndim == 2

    radiance = torch.fft.rfft(radiance)
    delay = torch.fft.rfft(delay_signal)
    transfered_radiance = radiance * delay
    transfered_radiance = torch.view_as_real(transfered_radiance)
    transfered_radiance = transfered_radiance.view(n, f * 2)
    transfered_radiance = reflection_kernel.matmul(transfered_radiance)
    transfered_radiance = transfered_radiance.view(n, f, 2)
    transfered_radiance = torch.view_as_complex(transfered_radiance)
    transfered_radiance = torch.fft.irfft(transfered_radiance)
    return transfered_radiance
