# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


def configs_arguments(parser):
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--pin_memory", action="store_false")
    parser.add_argument("--seed", default=4562, type=int)
    parser.add_argument("--task", default="DET", choices=["DET", "SEG"])

    parser.add_argument(
        "--dataset_name",
        choices=["mmvr"],
        type=str,
        default="mmvr",
    )
    parser.add_argument("--refined_ibox", action="store_true")

    parser.add_argument(
        "--split",
        type=str,
        default="P2S1",
        choices=["P1S1", "P1S2", "P2S1", "P2S2"],
        help="dataset split in [P1S1, P1S2, P2S1, P2S2] for MMVR",
    )
    parser.add_argument("--root", type=str, help="path to dataset")

    parser.add_argument(
        "--model_name",
        choices=["retr"],
        type=str,
        default="retr",
    )

    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--det_path",
        type=str,
        default=None,
        help="path of pre-trained detection model for segmentation",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"../logs/refined/dataset_name/split/task/",
        help="path name where to save checkpoint",
    )
    return parser


def optimizer_arguments(parser):
    parser.add_argument("--lr", default=0.0001, type=float, help="e.g., 0.000025")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument(
        "--save_every",
        default=1,
        type=int,
        help="eval and save the best checkpoint for each n epochs",
    )

    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_factor", default=None, type=int)
    parser.add_argument("--warmup_iters", default=None, type=int)
    parser.add_argument("--lr_drop", default=None, type=int)
    parser.add_argument("--worker", default=0, type=int, help="the number of workers")
    return parser
