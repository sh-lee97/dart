r"""
Calculations of reverberation parameters and its distances.
Implementation borrowed from other acoustic field codes (INRAS, AVR, ...) to match the implementation exactly.
"""
import numpy as np
import torch
from scipy import stats


def compare_echogram_parameters(
    gt_echogram, pred_echogram, radiance_sampling_rate, post=None, pre=None,
):
    try:
        gt_params = compute_echogram_parameters(gt_echogram, radiance_sampling_rate)
        pred_params = compute_echogram_parameters(pred_echogram, radiance_sampling_rate)
        distance = {}
        for key in gt_params:
            if key == "t60":
                distance[key] = 100 * np.mean(
                    np.abs(gt_params[key] - pred_params[key]) / gt_params[key]
                )
            elif key == "edt":
                distance[key] = (
                    np.mean(np.abs(gt_params[key] - pred_params[key]) / gt_params[key])
                    * 1000
                )
            elif key in ["echogram"]:
                distance[key] = np.mean(
                    np.abs(gt_params[key] - pred_params[key]) / gt_params[key].mean()
                )
            else:
                distance[key] = np.mean(np.abs(gt_params[key] - pred_params[key]))

        if pre is not None:
            distance = {f"{pre}/{k}": v for k, v in distance.items()}
        if post is not None:
            distance = {f"{k}_{post}": v for k, v in distance.items()}
        return distance
    except:
        return {}


def compute_echogram_parameters(echogram, radiance_sampling_rate):
    if isinstance(echogram, torch.Tensor):
        echogram = echogram.detach().cpu().numpy()
    if echogram.ndim == 1:
        echogram = echogram[None, :]

    boundary = int(0.05 * radiance_sampling_rate)
    e_early = np.sum(echogram[:, :boundary], -1)
    e_late = np.sum(echogram[:, boundary:], -1)
    c50 = 10.0 * np.log10(e_early / e_late)

    edc = np.cumsum(echogram[:, ::-1], axis=-1)[:, ::-1]
    edc = edc / edc[:, 0]
    edc = edc.clip(1e-8, 1)
    edc_log = 10 * np.log10(edc)

    t60, edt = t60_EDT_cal(edc_log, fs=radiance_sampling_rate)

    return dict(echogram=echogram, t60=t60, edt=edt, c50=c50)


def t60_EDT_cal(energys, init_db=-5, end_db=-25, factor=3.0, fs=48000):
    t60_all = []
    edt_all = []

    for energy in energys:
        # find the -10db point
        edt_factor = 6.0
        energy_n10db = energy[np.abs(energy - (-10)).argmin()]

        n10db_sample = np.where(energy == energy_n10db)[0][0]
        edt = n10db_sample / fs * edt_factor

        # find the intersection of -5db and -25db position
        energy_init = energy[np.abs(energy - init_db).argmin()]
        energy_end = energy[np.abs(energy - end_db).argmin()]
        init_sample = np.where(energy == energy_init)[0][0]
        end_sample = np.where(energy == energy_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = energy[init_sample : end_sample + 1]

        # regress to find the db decay trend
        slope, intercept = stats.linregress(x, y)[0:2]
        db_regress_init = (init_db - intercept) / slope
        db_regress_end = (end_db - intercept) / slope

        # get t60 value
        t60 = factor * (db_regress_end - db_regress_init)

        t60_all.append(t60)
        edt_all.append(edt)

    t60_all = np.array(t60_all)
    edt_all = np.array(edt_all)

    return t60_all, edt_all

