# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .parameters_mmvr import _NORMALIZE


class MMVR(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        type: str,
        data_split_path: str,
        last_frame_only: bool = True,
        normalize: bool = True,
        log_scale: bool = True,
        return_radar: bool = True,
        return_bbox: bool = True,
        return_mask: bool = True,
        return_pose: bool = True,
    ) -> None:
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split
        self.type = type
        self.data_split_path = data_split_path
        self.last_frame_only = last_frame_only
        self.normalize = normalize
        self.log_scale = log_scale
        self.return_radar = return_radar
        self.return_bbox = return_bbox
        self.return_mask = return_mask
        self.return_pose = return_pose

        self._sessions = self._get_sessions()
        self.file_list = self._get_file_list(self._sessions)

    def __len__(self) -> int:
        return len(self.file_list)

    def _get_sessions(self) -> list:
        data_split_data = np.load(self.data_split_path, allow_pickle=True)["data_split_dict"][0]
        data_split = data_split_data[self.split][self.type]

        return data_split

    def _get_file_list(self, sessions: list) -> list:
        file_list = []
        for s in sessions:
            file_list.extend(glob(os.path.join(self.dataset_path, s, "*_meta.npz")))
        return file_list

    def __getitem__(self, idx) -> Tensor:
        meta_path = Path(self.file_list[idx])
        parent_path, file_id = meta_path.parent, meta_path.stem.split("_")[0]
        item_dict = dict()
        item_dict["file_id"] = "_".join(os.path.join(parent_path, f"{file_id}").replace("\\", "/").split("/")[-4:])
        if self.return_radar:
            with np.load(os.path.join(parent_path, f"{file_id}_radar.npz")) as radar_data:
                item_dict["hm_hori"] = radar_data["hm_hori"]
                item_dict["hm_vert"] = radar_data["hm_vert"]
            # parameter
            day = next(
                re.search(r"d(\d+)s", part).group(1) for part in parent_path.parts if re.search(r"d(\d+)s", part)
            )
            if self.log_scale:
                m_hori, s_hori, nan_hori = _NORMALIZE[day]["hori"]["log"]
                m_vert, s_vert, nan_vert = _NORMALIZE[day]["vert"]["log"]
            else:
                m_hori, s_hori, nan_hori = _NORMALIZE[day]["hori"]["ori"]
                m_vert, s_vert, nan_vert = _NORMALIZE[day]["vert"]["ori"]

            # log-scale
            if self.log_scale:
                item_dict["hm_hori"] = np.log(item_dict["hm_hori"])
                item_dict["hm_vert"] = np.log(item_dict["hm_vert"])

            # nan_to_num
            item_dict["hm_hori"] = np.nan_to_num(item_dict["hm_hori"], nan=nan_hori)
            item_dict["hm_vert"] = np.nan_to_num(item_dict["hm_vert"], nan=nan_vert)
            # normalize
            if self.normalize:
                item_dict["hm_hori"] = (item_dict["hm_hori"] - m_hori) / s_hori
                item_dict["hm_vert"] = (item_dict["hm_vert"] - m_vert) / s_vert

            item_dict["hm_hori"] = torch.from_numpy(item_dict["hm_hori"])
            item_dict["hm_vert"] = torch.from_numpy(item_dict["hm_vert"])

            if len(item_dict["hm_hori"].shape) == 2:  # [height, width]
                item_dict["hm_hori"] = item_dict["hm_hori"].unsqueeze(0).unsqueeze(1)
                item_dict["hm_vert"] = item_dict["hm_vert"].unsqueeze(0).unsqueeze(1)
            elif len(item_dict["hm_hori"].shape) == 3:  # [seq_len, height, width]
                item_dict["hm_hori"] = item_dict["hm_hori"].unsqueeze(1)
                item_dict["hm_vert"] = item_dict["hm_vert"].unsqueeze(1)

            item_dict["env"] = torch.as_tensor(int(day), dtype=torch.int)

        if self.return_bbox:
            with np.load(os.path.join(parent_path, f"{file_id}_bbox.npz")) as bbox_data:
                item_dict["bbox_i"] = torch.from_numpy(bbox_data["bbox_i"]).to(torch.float)
                item_dict["bbox_hori"] = torch.from_numpy(bbox_data["bbox_hori"]).to(torch.float)
                item_dict["bbox_vert"] = torch.from_numpy(bbox_data["bbox_vert"]).to(torch.float)
            if self.last_frame_only:
                item_dict["bbox_i"] = item_dict["bbox_i"][-1].unsqueeze(0)
                item_dict["bbox_hori"] = item_dict["bbox_hori"][-1].unsqueeze(0)
                item_dict["bbox_vert"] = item_dict["bbox_vert"][-1].unsqueeze(0)
        if self.return_mask:
            with np.load(os.path.join(parent_path, f"{file_id}_mask.npz")) as mask_data:
                item_dict["mask"] = torch.from_numpy(mask_data["mask"]).to(torch.float)
            if self.last_frame_only:
                item_dict["mask"] = item_dict["mask"][-1].unsqueeze(0)
        if self.return_pose:
            pose_data = np.load(os.path.join(parent_path, f"{file_id}_pose.npz"))
            item_dict["kp"] = torch.from_numpy(pose_data["kp"]).to(torch.float)
            if self.last_frame_only:
                item_dict["kp"] = item_dict["kp"][-1].unsqueeze(0)

        return item_dict
