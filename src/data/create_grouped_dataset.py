# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy import ndarray

DATASET_PREFIX = ["P1", "P2"]
FILE_TYPES = ["meta", "radar", "mask", "bbox", "pose"]

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset_dir",
    type=str,
    help="str: The dir path including dataset (P1, P2), e.g., /home/foo/MMVR",
)
parser.add_argument(
    "-n",
    "--num_frames",
    type=int,
    help="int: The number of frames to be binded as a segment.",
)
parser.add_argument(
    "-o",
    "--overlap",
    type=int,
    default=0,
    help="int: The number of frames overlapping between segments. Default: 0 (non-overlap)",
)

parser.add_argument(
    "--output",
    type=str,
    default=".",
    help="output folder",
)
args = parser.parse_args()

assert args.num_frames > args.overlap, "--num_frames must be greater than overlap"


def get_file_lists(session_path: Path) -> dict:
    return {file_type: sorted(session_path.glob(f"*_{file_type}.npz")) for file_type in FILE_TYPES}


def annot_pad(annotation_list: list) -> ndarray:
    max_num = max(i.shape[0] for i in annotation_list)
    padded_annot = np.zeros((len(annotation_list), max_num, *annotation_list[0].shape[1:]))
    for idx, elem in enumerate(annotation_list):
        padded_annot[idx, : elem.shape[0]] = elem
    return padded_annot


def process_group(args):
    k, v, save_dir_path, num_frames, overlap = args
    for idx in range(0, len(v), num_frames - overlap):
        save_file_name = save_dir_path / f"{idx // (num_frames - overlap):05d}_{k}.npz"
        seg_list = []
        for i in range(idx - (num_frames - 1), min(idx + 1, len(v))):
            if i < 0:
                ind = idx
            else:
                ind = max(0, i)
            seg_list.append(np.load(str(v[ind])))
        seg_list.extend([seg_list[-1]] * (num_frames - len(seg_list)))

        if k == "meta":
            np.savez_compressed(
                save_file_name,
                global_frame_id=[str(s["global_frame_id"]) for s in seg_list],
            )
        elif k == "radar":
            np.savez_compressed(
                save_file_name,
                hm_hori=[s["hm_hori"] for s in seg_list],
                hm_vert=[s["hm_vert"] for s in seg_list],
            )
        elif k == "mask":
            np.savez_compressed(
                save_file_name,
                mask=[s["mask"] for s in seg_list][-1][None],
            )
        elif k == "bbox":
            np.savez_compressed(
                save_file_name,
                bbox_i=[s["bbox_i"] for s in seg_list][-1][None],
                bbox_hori=[s["bbox_hori"] for s in seg_list][-1][None],
                bbox_vert=[s["bbox_vert"] for s in seg_list][-1][None],
            )
        elif k == "pose":
            np.savez_compressed(
                save_file_name,
                kp=annot_pad([s["kp"] for s in seg_list]),
            )


def group_and_save(file_lists: dict, save_dir_path: Path, num_frames: int, overlap: int) -> None:
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_group, (k, v, save_dir_path, num_frames, overlap)) for k, v in file_lists.items()
        ]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    path_list = [Path(args.dataset_dir) / pre for pre in DATASET_PREFIX]
    save_prefix = Path(f"{args.output}/segment_{args.num_frames}_{args.overlap}")

    for p in path_list:
        session_list = sorted(p.glob("d*/*"))
        for s in session_list:
            print(f"Processing {s}...")
            # create save directory
            save_dir_path = save_prefix / s.relative_to(args.dataset_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)

            file_lists = get_file_lists(s)
            group_and_save(file_lists, save_dir_path, args.num_frames, args.overlap)


if __name__ == "__main__":
    main()
