# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import gc
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataloader import collate_det_seg, get_dataloader
from data.det_seg_dataset import MMVRDetSeg
from utils.common import EarlyStopping, get_model, get_optimizer, move_to_device, write_args
from utils.configs import *
from utils.detection_process import Metrics, plot_result, write_loss, write_metric


def get_args_parser():
    parser = argparse.ArgumentParser("set RETR training", add_help=False)
    parser = configs_arguments(parser)
    parser = optimizer_arguments(parser)
    return parser


def train_iter(
    output_dir,
    total_iter,
    epoch,
    train_loader,
    optimizer,
    scheduler,
    model,
    metrics_train,
    writer,
    device,
):
    running_loss = {}
    model.train()
    seg = True if args.task == "SEG" else False
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad(set_to_none=True)

        batch = move_to_device(batch, device)
        rf_hor = batch["hm_hori"].detach()
        rf_ver = batch["hm_vert"].detach()
        labels = batch["labels"]

        out, loss_dict = model(rf_hor, rf_ver, labels)

        loss = sum(loss_dict[k] for k in loss_dict.keys())
        loss.backward()
        optimizer.step()
        running_loss = {
            key: running_loss.get(key, 0) + loss_dict.get(key, 0).item() for key in set(running_loss) | set(loss_dict)
        }
        metrics_train.compute(labels, out)

        total_iter += 1
        del loss_dict, loss, labels

    scheduler.step()
    running_loss = {k: v / len(train_loader) for k, v in running_loss.items()}
    total_loss = sum(running_loss[k] for k in running_loss.keys())
    for k, v in running_loss.items():
        writer.add_scalar("Train/" + k, v, total_iter)
    writer.add_scalar("Train_loss/total_loss", total_loss, total_iter)
    writer.add_scalar("epoch", epoch, total_iter)
    writer.add_scalar("Train_loss/lr", optimizer.param_groups[0]["lr"], total_iter)

    metrics_train.get_result()
    write_loss(output_dir, total_loss, running_loss, epoch, "Train")
    write_metric(output_dir, metrics_train, writer, epoch, total_iter, "Train", seg=seg)
    plot_result(
        output_dir,
        rf_hor[0],
        rf_ver[0],
        batch["labels"][0]["iboxes"],
        batch["labels"][0]["hboxes"],
        batch["labels"][0]["vboxes"],
        batch["labels"][0]["masks"],
        out[0]["iboxes"].detach(),
        out[0]["hboxes"].detach(),
        out[0]["vboxes"].detach(),
        out[0]["masks_person"].detach() if args.task == "SEG" else None,
        out[0]["scores"].detach(),
        save_name=f"Train_epoch{str(epoch).zfill(3)}",
    )
    del out, batch, rf_hor, rf_ver, running_loss, total_loss
    gc.collect()
    torch.cuda.empty_cache()
    return total_iter, model


def val_test_iter(
    best_epoch,
    output_dir,
    total_iter,
    epoch,
    val_loader,
    model,
    early_stopping,
    stop,
    min_val_loss,
    metrics_val,
    writer,
    device,
):
    with torch.no_grad():
        model.eval()
        for loader, metrics, mode in zip([val_loader], [metrics_val], ["Val"]):
            seg = True if args.task == "SEG" else False
            print(f"[Epoch {epoch}: {mode}]")
            running_loss = {}
            for j, batch in enumerate(tqdm(loader)):
                batch = move_to_device(batch, device)
                rf_hor = batch["hm_hori"].detach()
                rf_ver = batch["hm_vert"].detach()
                labels = batch["labels"]

                out, loss_dict = model(rf_hor, rf_ver, labels)
                running_loss = {
                    key: running_loss.get(key, 0) + loss_dict.get(key, 0).item()
                    for key in set(running_loss) | set(loss_dict)
                }
                metrics.compute(labels, out)
                del loss_dict, labels

            metrics.get_result()
            write_metric(output_dir, metrics, writer, epoch, total_iter, mode, seg=seg)
            running_loss = {k: v / len(loader) for k, v in running_loss.items()}
            total_loss = sum(running_loss[k] for k in running_loss.keys())

            for k, v in running_loss.items():
                writer.add_scalar(f"{mode}_loss/" + k, v, total_iter)
            writer.add_scalar(f"{mode}_loss/total_loss", total_loss, total_iter)
            write_loss(output_dir, total_loss, running_loss, epoch, mode)

            if total_loss < min_val_loss and mode == "Val":
                torch.save(model.state_dict(), output_dir / f"best.pth")
                min_val_loss = total_loss
                best_epoch = epoch
                print(f"[Epoch {epoch}] model saved -> best.pth")
            if args.patience is not None and mode == "Val":  # early stopping
                stop = True if early_stopping(total_loss) else False

            plot_result(
                output_dir,
                rf_hor[0],
                rf_ver[0],
                batch["labels"][0]["iboxes"],
                batch["labels"][0]["hboxes"],
                batch["labels"][0]["vboxes"],
                batch["labels"][0]["masks"],
                out[0]["iboxes"].detach(),
                out[0]["hboxes"].detach(),
                out[0]["vboxes"].detach(),
                out[0]["masks_person"].detach() if seg else None,
                out[0]["scores"].detach(),
                save_name=f"{mode}_epoch{str(epoch).zfill(3)}",
            )
            del out, batch, rf_hor, rf_ver, running_loss, total_loss
            gc.collect()
            torch.cuda.empty_cache()
    return total_iter, best_epoch, stop, min_val_loss


def main(args):
    device = torch.device(args.device)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    writer = SummaryWriter(log_dir=args.output_dir)

    dataset_path = Path.joinpath(Path(args.root), args.split[:-2])

    train_loader, val_loader, _, args = get_dataloader(
        MMVRDetSeg,
        dataset_path,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.worker,
        collate_fn=collate_det_seg,
        pin_memory=args.pin_memory,
        cfg=args,
    )

    model = get_model(args, device)
    optimizer, scheduler = get_optimizer(args, model, len(train_loader))
    early_stopping = EarlyStopping(args.patience)

    seg = True if args.task == "SEG" else False
    metrics_train = Metrics(seg=seg).to(device)
    metrics_val = Metrics(seg=seg).to(device)
    output_dir = Path(args.output_dir)
    write_args(output_dir, args)

    print("\nStart training...")
    start_time = time.time()
    min_val_loss, best_epoch = np.inf, 0
    total_iter, stop = 0, False
    for epoch in range(args.epochs):
        print(f"[Epoch {epoch}: Training]")
        if stop:
            break

        """ Training """
        total_iter, model = train_iter(
            output_dir,
            total_iter,
            epoch,
            train_loader,
            optimizer,
            scheduler,
            model,
            metrics_train,
            writer,
            device,
        )

        """ Validation """
        if (epoch + 1) % args.save_every == 0:
            total_iter, best_epoch, stop, min_val_loss = val_test_iter(
                best_epoch,
                output_dir,
                total_iter,
                epoch,
                val_loader,
                model,
                early_stopping,
                stop,
                min_val_loss,
                metrics_val,
                writer,
                device,
            )

    total_time = time.time() - start_time
    print(f"Training time {total_time} [sec]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("RETR training script", parents=[get_args_parser()])
    args = parser.parse_args()
    task = "DET" if args.task == "DET" else "DETSEG"
    args.output_dir = args.output_dir.replace("/task/", f"/{task}/")
    args.output_dir = args.output_dir.replace("/dataset_name/", f"/{args.dataset_name}/")
    args.output_dir = args.output_dir.replace("/split/", f"/{args.split}/")
    args.output_dir = args.output_dir + datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
