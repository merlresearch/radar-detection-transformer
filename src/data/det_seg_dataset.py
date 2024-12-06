# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
from torch import Tensor

from .common import _resize, add_offset
from .mmvr_dataset import MMVR


class MMVRDetSeg(MMVR):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        type: str,
        refine: bool = False,
    ) -> None:
        super().__init__(
            dataset_path,
            split,
            type,
            "./utils/data_split.npz",
            True,
            True,
            True,
            True,
            True,
            True,
            False,
        )

    def __len__(self) -> int:
        return len(self.file_list)

    def annot_pad(self, data, n_sbj, coef=0, pad_max=5):
        n_dim = data.ndim
        if n_dim >= 2:
            return torch.cat((data, torch.zeros((pad_max - n_sbj, *data.shape[1:]))), dim=0)
        elif n_dim == 1:
            return torch.cat((data, torch.zeros(pad_max - n_sbj) + coef))

    def __getitem__(self, idx) -> Tensor:
        item_dict = super().__getitem__(idx)

        item_dict["hm_hori"] = item_dict["hm_hori"].permute(1, 0, 2, 3)[0]
        item_dict["hm_vert"] = item_dict["hm_vert"].permute(1, 0, 2, 3)[0]

        item_dict["n_sbj"] = n_sbj = item_dict["bbox_i"].shape[1]
        item_dict["bbox_i"] = self.annot_pad(item_dict["bbox_i"][0][:, :4] / 2, n_sbj)
        item_dict["bbox_hori"] = self.annot_pad(item_dict["bbox_hori"][0], n_sbj)
        item_dict["bbox_vert"] = self.annot_pad(item_dict["bbox_vert"][0], n_sbj)
        item_dict["mask"] = self.annot_pad(_resize(item_dict["mask"][0], 0.5), n_sbj)
        item_dict["labels"] = self.annot_pad(torch.zeros(n_sbj), n_sbj, -1)

        item_dict = add_offset(item_dict, "hori", 5.0, 10.0)
        item_dict = add_offset(item_dict, "vert", 5.0, 10.0)

        return item_dict
