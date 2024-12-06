# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from torchmetrics import JaccardIndex
from torchmetrics.detection import MeanAveragePrecision


class Metrics(nn.Module):
    def __init__(self, seg=False) -> None:
        super().__init__()

        """ define metrics """
        self.seg = seg
        if self.seg:
            self.iou_image = nn.ModuleDict(
                {
                    "iou": JaccardIndex(task="multiclass", num_classes=2),
                }
            )
        self.map_image = nn.ModuleDict(
            {
                "mAP": MeanAveragePrecision(iou_type="bbox", box_format="xyxy"),
            }
        )
        self.map_radar = nn.ModuleDict(
            {
                "mAP": MeanAveragePrecision(
                    iou_type="bbox",
                    box_format="xyxy",
                ),
            }
        )
        self.res = {}

    def compute(self, gt, out):
        if self.seg:
            predictions = []
            gt_masks = []
            for prediction, target in zip(out, gt):
                current_gt = torch.where(target["masks"].sum(0) > 0, 1, 0)
                gt_masks.append(current_gt)
                predictions.append(prediction["masks"])

            """ record image plane iou scores """
            predictions = torch.stack(predictions)
            gt_masks = torch.stack(gt_masks)
            self.iou_image["iou"].update(predictions, gt_masks)

        """ record image plane bbox scores """
        preds = [
            {
                "boxes": elem["iboxes"],
                "scores": elem["scores"],
                "labels": elem["labels"].long(),
            }
            for elem in out
        ]
        targets = [
            {
                "boxes": elem["iboxes"][: elem["n_sbj"]],
                "labels": elem["labels"][: elem["n_sbj"]].long(),
            }
            for elem in gt
        ]
        self.map_image["mAP"].update(preds, targets)

        """ record radar plane bbox scores """
        preds_radar = [
            {
                "boxes": elem["hboxes"],
                "scores": elem["scores"],
                "labels": elem["labels"].long(),
            }
            for elem in out
        ]
        targets_radar = [
            {
                "boxes": elem["hboxes"][: elem["n_sbj"]],
                "labels": elem["labels"][: elem["n_sbj"]].long(),
            }
            for elem in gt
        ]
        self.map_radar["mAP"].update(preds_radar, targets_radar)

    def get_result(self):
        print("Image Plane:")
        if self.seg:
            res_iou = self.iou_image["iou"].compute()
            print("IoU:    ", res_iou.item())
            self.iou_image["iou"].reset()
        res_image = self.map_image["mAP"].compute()
        print("AP:    ", res_image["map"].item())
        print("AP_50: ", res_image["map_50"].item())
        print("AP_75: ", res_image["map_75"].item())
        print("AR_1: ", res_image["mar_1"].item())
        print("AR_10: ", res_image["mar_10"].item())
        self.map_image["mAP"].reset()
        res_radar = self.map_radar["mAP"].compute()
        self.map_radar["mAP"].reset()

        self.res = {
            "det_img": res_image,
            "det_rad": res_radar,
            "seg_img": res_iou if self.seg else None,
        }


def write_loss(output_dir, total_loss, running_loss, epoch, mode="Val"):
    with (output_dir / "log.txt").open("a") as f:
        f.write(
            f"[Epoch {epoch:04d}] {mode}: loss {total_loss},"
            f"{''.join(f'{key}:{value},' for key, value in running_loss.items())}\n"
        )


def write_metric(output_dir, metrics, writer, epoch, total_iter, mode="Val", seg=False):
    with (output_dir / "log.txt").open("a") as f:
        f.write(
            f"[Epoch {epoch:04d}] {mode}: "
            f"mAP {metrics.res['det_img']['map'].item()}, "
            f"mAP50 {metrics.res['det_img']['map_50'].item()}, "
            f"mAP75 {metrics.res['det_img']['map_75'].item()}, "
            f"mAR1 {metrics.res['det_img']['mar_1'].item()}, "
            f"mAR10 {metrics.res['det_img']['mar_10'].item()}, "
            f"IoU {metrics.res['seg_img'].item() if seg else ''}\n"
        )

    writer.add_scalar(f"{mode}_image/mAP", metrics.res["det_img"]["map"].item(), total_iter)
    writer.add_scalar(f"{mode}_image/mAP50", metrics.res["det_img"]["map_50"].item(), total_iter)
    writer.add_scalar(f"{mode}_image/mAP75", metrics.res["det_img"]["map_75"].item(), total_iter)
    writer.add_scalar(f"{mode}_image/mAR1", metrics.res["det_img"]["mar_1"].item(), total_iter)
    writer.add_scalar(f"{mode}_image/mAR10", metrics.res["det_img"]["mar_10"].item(), total_iter)
    writer.add_scalar(f"{mode}_image/IoU", metrics.res["seg_img"].item(), total_iter) if seg else None

    writer.add_scalar(f"{mode}_radar/mAP", metrics.res["det_rad"]["map"].item(), total_iter)
    writer.add_scalar(f"{mode}_radar/mAP50", metrics.res["det_rad"]["map_50"].item(), total_iter)
    writer.add_scalar(f"{mode}_radar/mAP75", metrics.res["det_rad"]["map_75"].item(), total_iter)
    writer.add_scalar(f"{mode}_radar/mAR1", metrics.res["det_rad"]["mar_1"].item(), total_iter)
    writer.add_scalar(f"{mode}_radar/mAR10", metrics.res["det_rad"]["mar_10"].item(), total_iter)


def plot_result(
    output_dir,
    rf_hor,
    rf_ver,
    gtbbox_i,
    gtbbox_hori,
    gtbbox_vert,
    gtmask,
    prbbox_i,
    prbbox_hori,
    prbbox_vert,
    prmask,
    scores,
    save_name="result",
):
    rf_hor, rf_ver = rf_hor.cpu().numpy(), rf_ver.cpu().numpy()
    gtbbox_i, gtbbox_hori, gtbbox_vert, gtmask = (
        gtbbox_i.cpu().numpy(),
        gtbbox_hori.cpu().numpy(),
        gtbbox_vert.cpu().numpy(),
        gtmask.cpu().numpy(),
    )
    prbbox_i, prbbox_hori, prbbox_vert, prmask = (
        prbbox_i.cpu().numpy(),
        prbbox_hori.cpu().numpy(),
        prbbox_vert.cpu().numpy(),
        prmask.cpu().numpy() if prmask is not None else None,
    )
    scores = scores.cpu().numpy()

    figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    figure.tight_layout()
    for ax, canvas, gt, pr, title, score in zip(
        [ax1, ax2, ax3],
        [rf_hor[-1], rf_ver[-1], np.sum(gtmask, axis=0)],
        [gtbbox_hori, gtbbox_vert, gtbbox_i],
        [prbbox_hori, prbbox_vert, prbbox_i],
        ["hori", "vert", "image"],
        [scores, scores, scores],
    ):
        ax.imshow(canvas)
        for box in gt:
            x0, y0, x1, y1 = box
            rect = Rectangle(
                (x0, y0),
                (x1 - x0),
                (y1 - y0),
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)
        for i, box in enumerate(pr):
            x0, y0, x1, y1 = box
            rect = Rectangle(
                (x0, y0),
                (x1 - x0),
                (y1 - y0),
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            ax.text(
                x0 + 5,
                y0 + 5,
                f"{score[i]:.2f}",
                color="red",
                verticalalignment="top",
                fontsize=12,
                fontweight="bold",
            )
            ax.add_patch(rect)
        ax.set_title(title)
        ax.set_axis_off()
    if prmask is not None:
        colors = ["Reds", "Greens", "Blues", "Purples"]
        for mask, color in zip(prmask, colors):
            ax4.imshow(mask, cmap=color, alpha=0.6)
        ax4.set_title("masks")
        ax4.set_axis_off()
    plt.savefig(Path.joinpath(output_dir, f"{save_name}.png"))
    plt.close()
    gc.collect()
    torch.cuda.empty_cache()
