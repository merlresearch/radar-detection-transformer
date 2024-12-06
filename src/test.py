# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.dataloader import collate_det_seg, get_dataloader
from data.det_seg_dataset import MMVRDetSeg
from models import RETR
from utils.common import move_to_device
from utils.detection_process import Metrics

project_root = Path(__file__).resolve().parent.parent


def get_args_parser():
    parser = argparse.ArgumentParser("set RETR inference", add_help=False)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--worker", default=2, type=int, help="the number of workers")

    # dataset arguments
    parser.add_argument("--dataset_name", type=str, help="dataset name in data")
    parser.add_argument("--split", default="P2S1", type=str, help="dataset split")
    parser.add_argument("--root", type=str, help="path to dataset")

    # model
    parser.add_argument(
        "--pretrained_path",
        default="../logs/pretrained_model/p2s1_retr_detseg.pth",
        type=str,
        help="checkpoint path",
    )
    parser.add_argument("--task", default="DETSEG", type=str, choices=["DETSEG", "DET"])
    return parser


def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.task == "DETSEG":
        args.task = "SEG"
    model = RETR(task=args.task).to(device)
    params = torch.load(args.pretrained_path)
    model.load_state_dict(params)

    dataset_path = Path.joinpath(Path(args.root), args.split[:2])
    _, _, test_loader = get_dataloader(
        MMVRDetSeg,
        dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        collate_fn=collate_det_seg,
        num_workers=args.worker,
    )

    metrics = Metrics(True if args.task == "SEG" else False).to(device)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = move_to_device(batch, device)
            rf_hor = batch["hm_hori"].detach()
            rf_ver = batch["hm_vert"].detach()
            labels = batch["labels"]

            out = model(rf_hor, rf_ver)

            metrics.compute(labels, out)
        metrics.get_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RETR test script", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
