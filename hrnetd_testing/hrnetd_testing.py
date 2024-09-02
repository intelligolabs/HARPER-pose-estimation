import argparse
import os
from tqdm import tqdm
from pathlib import Path

import numpy as np
import math
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from lib import dataset
from lib.models import models
from lib.config import cfg
from lib.config import update_config
from lib.utils.vis import save_debug_images
from lib.core.loss import JointsMSELoss
from lib.core.function import AverageMeter
from lib.core.evaluate import accuracy
from lib.core.inference import get_max_preds

from hrnetd_testing_function import create_3d_projection, compute_3d_output


def validate(
    config, val_loader, val_dataset, model, criterion, output_dir, compute_3d=False
):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    estimated_3d_poses = []

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    image_path = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(tqdm(val_loader)):
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            batch_heatmaps = output.clone().cpu().numpy()
            coords, maxvals = get_max_preds(batch_heatmaps)
            heatmap_height = batch_heatmaps.shape[2]
            heatmap_width = batch_heatmaps.shape[3]

            # post-processing
            if config.TEST.POST_PROCESS:
                for ndx in range(coords.shape[0]):
                    for pdx in range(coords.shape[1]):
                        hm = batch_heatmaps[ndx][pdx]
                        px = int(math.floor(coords[ndx][pdx][0] + 0.5))
                        py = int(math.floor(coords[ndx][pdx][1] + 0.5))
                        if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                            diff = np.array(
                                [
                                    hm[py][px + 1] - hm[py][px - 1],
                                    hm[py + 1][px] - hm[py - 1][px],
                                ]
                            )
                            coords[ndx][pdx] += np.sign(diff) * 0.25

            resize_scale_x = meta["resize_scale_x"][:, None].numpy().astype(np.float32)
            resize_scale_y = meta["resize_scale_y"][:, None].numpy().astype(np.float32)
            # Transform the coordinates from the heatmap space to the original image space
            target_coords = coords.copy() * 4.0
            target_coords[:, :, 0] *= 1 / resize_scale_x
            target_coords[:, :, 1] *= 1 / resize_scale_y
            preds = target_coords

            all_preds[idx : idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 2:3] = maxvals
            image_path.extend(meta["image"])
            idx += num_images

            if compute_3d:
                estimated_3d_poses.extend(
                    create_3d_projection(
                        config,
                        preds,
                        meta,
                        output_dir,
                        visualize=config.DEBUG.DEBUG & config.DEBUG.SAVE_3D_PLOTS,
                    )
                )

            if i % config.PRINT_FREQ == 0:
                msg = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {acc.val:.3f} ({acc.avg:.3f})".format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, acc=acc
                    )
                )

                prefix = "{}_{}".format(os.path.join(output_dir, "val"), i)
                save_debug_images(config, input, meta, target, pred * 4, output, prefix)
    print(f"AVG ACC: {round(acc.avg, 2)}")  # PCK metric

    if compute_3d:
        compute_3d_output(config, output_dir, estimated_3d_poses)
    return acc.avg


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # general
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)
    root_output_dir = Path(cfg.OUTPUT_DIR)
    cfg_name = os.path.basename(args.cfg).split(".")[0]
    final_output_dir = root_output_dir / cfg.DATASET.DATASET / cfg.MODEL.NAME / cfg_name
    print("=> creating {}".format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        print("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        raise ValueError("Please specify the model file for testing!")

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    valid_dataset = eval("dataset." + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TEST_SET,
        False,
        transforms.Compose([transforms.ToTensor()]),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )

    # evaluate on validation set
    validate(
        cfg,
        valid_loader,
        valid_dataset,
        model,
        criterion,
        final_output_dir,
        compute_3d=True,
    )


if __name__ == "__main__":
    main()
