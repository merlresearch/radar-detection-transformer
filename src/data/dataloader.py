# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import List

import torch
from torch.utils.data import DataLoader


def collate_det_seg(batch: List) -> dict:
    batch_dict = dict()
    hm_hori_stack = [b["hm_hori"] for b in batch]
    hm_vert_stack = [b["hm_vert"] for b in batch]
    hm_hori = torch.stack(hm_hori_stack, dim=0)
    hm_vert = torch.stack(hm_vert_stack, dim=0)
    batch_dict["hm_hori"] = hm_hori
    batch_dict["hm_vert"] = hm_vert

    batch_dict["labels"] = [
        {
            "iboxes": b["bbox_i"],
            "hboxes": b["bbox_hori"],
            "vboxes": b["bbox_vert"],
            "masks": b["mask"],
            "n_sbj": b["n_sbj"],
            "labels": b["labels"],
            "env": b["env"],
            "file_id": b["file_id"],
        }
        for b in batch
    ]

    return batch_dict


def collate_pose(batch: List) -> dict:
    batch_dict = dict()
    # radar: normal collate
    hm_hori_stack = [b["hm_hori"] for b in batch]
    hm_vert_stack = [b["hm_vert"] for b in batch]
    hm_hori = torch.stack(hm_hori_stack, dim=0)
    hm_vert = torch.stack(hm_vert_stack, dim=0)
    batch_dict["hm_hori"] = hm_hori
    batch_dict["hm_vert"] = hm_vert

    # pose: heatmap -> max, paf -> sum
    hm_pose_stack = [b["heatmap"] for b in batch]
    paf_stack = [b["paf"] for b in batch]
    hm_pose = torch.stack(hm_pose_stack, dim=0)
    paf = torch.stack(paf_stack, dim=0)
    batch_dict["heatmap"] = hm_pose
    batch_dict["paf"] = paf

    # keypoint
    kp_stack = [b["kp"] for b in batch]
    kp = torch.stack(kp_stack, dim=0)
    batch_dict["kp"] = kp

    return batch_dict


def get_dataloader(
    Dataset,
    dataset_path,
    split: str,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn=None,
    cfg=None,
    pin_memory=False,
):
    train_dataset = Dataset(
        dataset_path=dataset_path,
        split=split,
        type="train",
    )
    val_dataset = Dataset(
        dataset_path=dataset_path,
        split=split,
        type="val",
    )
    test_dataset = Dataset(
        dataset_path=dataset_path,
        split=split,
        type="test",
    )

    # return dataloader using custom collate fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    if cfg is None:
        return train_loader, val_loader, test_loader
    else:
        cfg.num_train = train_dataset.__len__()
        cfg.num_val = val_dataset.__len__()
        cfg.num_test = test_dataset.__len__()
        cfg.sessions_train = "".join([s + ", " for s in train_dataset._sessions])
        cfg.sessions_val = "".join([s + ", " for s in val_dataset._sessions])
        cfg.sessions_test = "".join([s + ", " for s in test_dataset._sessions])
        return train_loader, val_loader, test_loader, cfg
