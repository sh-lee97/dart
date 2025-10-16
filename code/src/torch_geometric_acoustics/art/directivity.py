r"""
Learnable directivity module. 
"""

import torch
import torch.nn as nn


class LearnableAxialDirectivity(nn.Module):
    def __init__(self, n_bins=128, sharpness=8):
        super().__init__()

        angle_points = torch.linspace(0, torch.pi, n_bins)
        self.register_buffer("points", angle_points)
        magnitude = torch.ones(n_bins)
        self.magnitude = nn.Parameter(magnitude)
        self.sharpness = sharpness

    def forward(self, directions, orientation):
        r"""
        directions: (N, 3)
        orientation_direction: (3)
        """
        orientation = orientation[0]
        dots = (directions * orientation[None, :]).sum(-1)
        dots = torch.clamp(dots, -1 + 1e-3, 1 - 1e-3)
        angles = torch.acos(dots)

        distance = torch.abs(angles[:, None] - self.points[None, :])
        weights = torch.softmax(-self.sharpness * distance, -1)
        magnitude = self.magnitude[None, :]  
        response = (weights * magnitude).sum(-1)
        return response