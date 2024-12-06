# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, SequentialLR

from models import RETR


def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    else:
        return batch


def recursive_detach(tensor_or_dict):
    if isinstance(tensor_or_dict, dict):
        return {k: recursive_detach(v) for k, v in tensor_or_dict.items()}
    elif isinstance(tensor_or_dict, list):
        return [recursive_detach(v) for v in tensor_or_dict]
    elif isinstance(tensor_or_dict, torch.Tensor):
        return tensor_or_dict.detach()
    else:
        return tensor_or_dict


def write_args(output_dir, cfg):
    with (output_dir / "log.txt").open("a") as f:
        [f.write(f"{k}: {v}\n") for (k, v) in cfg._get_kwargs()]
    [print(f"{k}: {v}") for (k, v) in cfg._get_kwargs()]


def get_optimizer(args, model, steps_per_epoch=None):
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.scheduler == "liner":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.1, args.lr_drop)
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
    elif args.scheduler == "cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
        )
    else:
        raise NotImplementedError

    if args.warmup:
        warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_factor, total_iters=args.warmup_iters)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[args.warmup_iters],
        )

    return optimizer, scheduler


def get_model(args, device):
    if args.model_name == "retr":
        model = RETR(task=args.task, path=args.det_path)
    else:
        raise NotImplementedError

    model = model.to(device)

    if args.resume_path is not None:
        model.load_state_dict(torch.load(args.resume_path))
    return model


class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = True):
        self.epoch = 0
        self.pre_loss = float("inf")
        self.patience = patience
        self.verbose = verbose

    def __call__(self, current_loss):
        if self.pre_loss < current_loss:
            self.epoch += 1
            if self.epoch > self.patience:
                if self.verbose:
                    print("early stopping")
                return True
        else:
            self.epoch = 0
            self.pre_loss = current_loss

        return False
