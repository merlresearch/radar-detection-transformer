# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch


def _resize(image, scale_factor=0.5):
    dim = image.ndim
    if dim == 2:
        image = image[None, None]  # .clone().detach()
    elif dim == 3:
        image = image[None]  # .clone().detach()
    resized = torch.nn.functional.interpolate(image, scale_factor=scale_factor, mode="nearest")[0]
    resized = (resized >= 0.5).long()  # .clone().detach()
    return resized


def add_offset(data, type: str = "hori", offset_tl: float = 0.1, offset_br: float = 0.1):
    n_sbj = data["n_sbj"]
    data["bbox_" + type][:n_sbj, :2] = data["bbox_" + type][:n_sbj, :2] - offset_tl
    data["bbox_" + type][:n_sbj, 2:] = data["bbox_" + type][:n_sbj, 2:] + offset_br
    data["bbox_" + type][:n_sbj, :2][data["bbox_" + type][:n_sbj, :2] < 0] = 0
    data["bbox_" + type][:n_sbj, 2][data["bbox_" + type][:n_sbj, 2] > data["hm_" + type].shape[2]] = data[
        "hm_" + type
    ].shape[2]
    data["bbox_" + type][:n_sbj, 3][data["bbox_" + type][:n_sbj, 3] > data["hm_" + type].shape[1]] = data[
        "hm_" + type
    ].shape[1]
    return data
