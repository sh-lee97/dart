r"""
Optimization objectives for DART.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LossHandler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_echogram,
        gt_echogram,
        echogram_loss_type=["mse", "edc"],
    ):
        loss = {}
        echogram_loss = 0

        if "mse" in echogram_loss_type:
            loss["echogram_loss_mse"] = F.mse_loss(pred_echogram, gt_echogram)
            loss["echogram_loss_mse"] = loss["echogram_loss_mse"] / (
                gt_echogram.square().mean(-1, keepdim=True) + 1e-5
            )
            loss["echogram_loss_mse"] = loss["echogram_loss_mse"].mean()
            echogram_loss = echogram_loss + loss["echogram_loss_mse"]  

        if "edc" in echogram_loss_type:
            pred_edc = torch.flip(pred_echogram, (-1,)).cumsum(-1)
            gt_edc = torch.flip(gt_echogram, (-1,)).cumsum(-1)
            loss["echogram_loss_edc"] = F.l1_loss(pred_edc, gt_edc)
            loss["echogram_loss_edc"] = loss["echogram_loss_edc"] / gt_edc.mean(
                -1, keepdim=True
            )
            loss["echogram_loss_edc"] = loss["echogram_loss_edc"].mean()
            echogram_loss = echogram_loss + loss["echogram_loss_edc"]  

        loss["echogram_loss"] = echogram_loss
        return loss
