# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import random
import re
from datetime import timedelta
from typing import Optional, no_type_check

import hydra

import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from torch import Tensor
from torchmetrics.classification import Dice
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod


def plot_and_save(targets_list, inputs_list, dir=f'sam2_logs/fig', prefix="image_", extension=".png", is_show=False, metadatas_list=None):
    return
    if not os.path.exists(dir):
        os.makedirs(dir)
    files = os.listdir(dir)
    matching_files = [f for f in files if f.startswith(prefix) and f.endswith(extension)]
    indices = []
    for file_name in matching_files:
        try:
            index = int(file_name[len(prefix):-len(extension)])  # 提取数字部分
            indices.append(index)
        except ValueError:
            pass
    idx = max(indices, default=0) + 1
    if metadatas_list is None:
        metadatas_list = torch.zeros(targets_list.shape[0], targets_list.shape[1], 3, dtype=torch.int32)

    for outs, targets, metadatas in zip(inputs_list, targets_list, metadatas_list):
        targets = targets.detach().cpu()
        pred_masks = outs['multistep_pred_multimasks_high_res'][0].detach().cpu()
        ious = outs['multistep_pred_ious'][0].detach().cpu()
        metadatas = metadatas.detach().cpu()

        for i in range(pred_masks.shape[0]):
            gt_mask = targets[i,:,:].squeeze(0)
            pred_mask = pred_masks[i, :, :, :].squeeze(0)
            iou = ious[i,:].squeeze(0).to(torch.float32).numpy()
            metadata = metadatas[i,:].squeeze(0).numpy()
            # 创建一个新的图形窗口
            fig, ax = plt.subplots(2, 2, figsize=(12, 12))  # 1 行 2 列的子图

            # 可视化 Ground Truth Mask
            ax[0,0].imshow(gt_mask.numpy(), cmap='Blues')  # 使用蓝色显示 GT Mask
            ax[0,0].set_title(f'Ground Truth Mask. Meta:{metadata}')  # 设置标题(video_id, obj_id, frame_id)
            ax[0,0].axis('off')  # 关闭坐标轴显示

            # 可视化 Prediction Mask
            ax[0,1].imshow(pred_mask[0,:,:].squeeze(0).numpy(), cmap='Reds')  # 使用红色显示预测 Mask
            ax[0,1].set_title(f'Prediction Mask IoU: {iou[0]:.4f}')  # 设置标题
            ax[0,1].axis('off')  # 关闭坐标轴显示
            # 可视化 Prediction Mask
            ax[1,0].imshow(pred_mask[1,:,:].squeeze(0).numpy(), cmap='Reds')  # 使用红色显示预测 Mask
            ax[1,0].set_title(f'Prediction Mask IoU: {iou[1]:.4f}')  # 设置标题
            ax[1,0].axis('off')  # 关闭坐标轴显示
            # 可视化 Prediction Mask
            ax[1,1].imshow(pred_mask[2,:,:].squeeze(0).numpy(), cmap='Reds')  # 使用红色显示预测 Mask
            ax[1,1].set_title(f'Prediction Mask IoU: {iou[2]:.4f}')  # 设置标题
            ax[1,1].axis('off')  # 关闭坐标轴显示

            plt.savefig(os.path.join(dir, f"{prefix}{idx}{extension}"))
            idx += 1
            if is_show:
                plt.show()
            plt.close(fig)


def multiply_all(*args):
    return np.prod(np.array(args)).item()


def collect_dict_keys(config):
    """This function recursively iterates through a dataset configuration, and collect all the dict_key that are defined"""
    val_keys = []
    # If the this config points to the collate function, then it has a key
    if "_target_" in config and re.match(r".*collate_fn.*", config["_target_"]):
        val_keys.append(config["dict_key"])
    else:
        # Recursively proceed
        for v in config.values():
            if isinstance(v, type(config)):
                val_keys.extend(collect_dict_keys(v))
            elif isinstance(v, omegaconf.listconfig.ListConfig):
                for item in v:
                    if isinstance(item, type(config)):
                        val_keys.extend(collect_dict_keys(item))
    return val_keys


class Phase:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("times", multiply_all)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    OmegaConf.register_new_resolver("pow", lambda x, y: x**y)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
    OmegaConf.register_new_resolver("range", lambda x: list(range(x)))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("ceil_int", lambda x: int(math.ceil(x)))
    OmegaConf.register_new_resolver("merge", lambda *x: OmegaConf.merge(*x))


def setup_distributed_backend(backend, timeout_mins):
    """
    Initialize torch.distributed and set the CUDA device.
    Expects environment variables to be set as per
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    along with the environ variable "LOCAL_RANK" which is used to set the CUDA device.
    """
    # enable TORCH_NCCL_ASYNC_ERROR_HANDLING to ensure dist nccl ops time out after timeout_mins
    # of waiting
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    logging.info(f"Setting up torch.distributed with a timeout of {timeout_mins} mins")
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_mins))
    return dist.get_rank()


def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", None))
    distributed_rank = int(os.environ.get("RANK", None))
    assert (
        local_rank is not None and distributed_rank is not None
    ), "Please the set the RANK and LOCAL_RANK environment variables."
    return local_rank, distributed_rank


def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    logging.info(OmegaConf.to_yaml(cfg))


def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    # Since in the pytorch sampler, we increment the seed by 1 for every epoch.
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"MACHINE SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        logging.info(f"Error creating directory: {dir_path}")
    return is_success


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_amp_type(amp_type: Optional[str] = None):
    if amp_type is None:
        return None
    assert amp_type in ["bfloat16", "float16"], "Invalid Amp type."
    if amp_type == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16


def log_env_variables():
    env_keys = sorted(list(os.environ.keys()))
    st = ""
    for k in env_keys:
        v = os.environ[k]
        st += f"{k}={v}\n"
    logging.info("Logging ENV_VARIABLES")
    logging.info(st)


def _iou_compute(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    average: Optional[str],
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    """Compute dice from the stat scores: true positives, false positives, false negatives.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        average: Defines the reduction that is applied
        mdmc_average: Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter)
        zero_division: The value to use for the score if denominator equals zero.
    """
    numerator = tp
    denominator = tp + fp + fn

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]

    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indices = torch.nonzero((tp | fn | fp) == 0).cpu()
        numerator[meaningless_indices, ...] = -1
        denominator[meaningless_indices, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=zero_division,
    )


class DiceMeter(Dice):
    def __init__(self, name="dice", **kwargs):
        super().__init__(sync_on_compute=True, **kwargs)
        self.name = name

    def is_better(self, a, b):
        return a >= b

    def update(self, outs_batch, targets_batch, **kwargs):
        assert len(outs_batch) == len(targets_batch)

        for outs, target_masks in zip(outs_batch, targets_batch):
            src_masks = outs["multistep_pred_masks_high_res"].squeeze(1)
            self._update(
                src_masks, target_masks
            )
            pass

    def _update(self, src_masks, target_masks):
        if src_masks.dim() == 4:
            src_masks = src_masks[:, -1, :, :].squeeze(1)
        assert src_masks.shape == target_masks.shape
        for src_mask, target_mask in zip(src_masks, target_masks):
            super().update(src_mask, target_mask)
        pass

    def compute(self):
        result = super().compute().item()
        return {self.name: result}

    def summarize(self):
        return self.compute()

class IoUMeter(DiceMeter):
    def __init__(self, name="iou", **kwargs):
        super().__init__(name=name, average='micro', mdmc_average='samplewise')
        self.add_state('categories', default=list(), dist_reduce_fx="cat")

    def update(self, outs_batch, targets_batch, **kwargs):
        assert len(outs_batch) == len(targets_batch)
        for outs, target_masks in zip(outs_batch, targets_batch):
            src_masks = outs["multistep_pred_masks_high_res"].squeeze(1)
            self._update(
                src_masks, target_masks
            )

    def compute(self):
        tp, fp, _, fn = self._get_final_stats()
        miou_result = self.miou_compute(tp, fp, fn).item()
        ciou_result = self.ciou_compute(tp, fp, fn).item()
        return {"mIoU": miou_result, "cIoU": ciou_result}

    @no_type_check
    def miou_compute(self, tp, fp, fn) -> Tensor:
        """Compute metric."""
        return _iou_compute(tp, fp, fn, 'micro', 'samplewise', self.zero_division)

    @no_type_check
    def ciou_compute(self, tp, fp, fn) -> Tensor:
        """Compute metric."""
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        return _iou_compute(tp, fp, fn, 'micro', 'global', self.zero_division)


class CateIoUMeter(DiceMeter):
    def __init__(self, name="iou", **kwargs):
        super().__init__(name=name, average='micro', mdmc_average='samplewise')
        self.add_state('categories', default=list(), dist_reduce_fx="cat")
        self.cates = {}
        self.catest = {}

    @no_type_check
    def update(self, outs_batch, targets_batch, categories_batch=None):
        assert len(outs_batch) == len(targets_batch)
        for outs, target_masks, categories in zip(outs_batch, targets_batch, categories_batch):
            for c in categories:
                # if c not in self.cates:
                #     self.cates[c] = len(self.cates)
                #     self.catest[len(self.catest)] = c
                # self.categories.append(torch.tensor(self.cates[c], device=self.device))
                self.categories.append(torch.tensor(c, device=self.device))
            src_masks = outs["multistep_pred_masks_high_res"].squeeze(1)
            self._update(
                src_masks, target_masks
            )

    def compute(self):
        tp, fp, _, fn = self._get_final_stats()
        category_result = {}
        for category in self.categories:
            label_tensor = torch.tensor([1 if category == c else 0 for c in self.categories], dtype=torch.bool)
            _tp, _fp, _fn = tp[label_tensor], fp[label_tensor], fn[label_tensor]
            _result = self.miou_compute(_tp, _fp, _fn).item()
            category_result[str(category.item())] = _result
        category_result['category_average'] = np.mean([v for v in category_result.values()])

        miou_result = self.miou_compute(tp, fp, fn).item()
        ciou_result = self.ciou_compute(tp, fp, fn).item()
        result = {"mIoU": miou_result, "cIoU": ciou_result}
        result.update(category_result)
        return result

    @no_type_check
    def summarize(self):
        tp, fp, _, fn = self._get_final_stats()
        category_result = {}
        for category in set(self.categories):
            label_tensor = torch.tensor([1 if category == c else 0 for c in self.categories], dtype=torch.bool)
            _tp, _fp, _fn = tp[label_tensor], fp[label_tensor], fn[label_tensor]
            _result = self.miou_compute(_tp, _fp, _fn).item()
            category_result[category.cpu().item()] = _result
        category_result['category_average'] = np.mean([v for v in category_result.values()])

        miou_result = self.miou_compute(tp, fp, fn).item()
        ciou_result = self.ciou_compute(tp, fp, fn).item()
        result = {"mIoU": miou_result, "cIoU": ciou_result}
        result.update(category_result)
        return result

    @no_type_check
    def miou_compute(self, tp, fp, fn) -> Tensor:
        """Compute metric."""
        return _iou_compute(tp, fp, fn, 'micro', 'samplewise', self.zero_division)

    @no_type_check
    def ciou_compute(self, tp, fp, fn) -> Tensor:
        """Compute metric."""
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        return _iou_compute(tp, fp, fn, 'micro', 'global', self.zero_division)


class PrecisionMeter(DiceMeter):
    def __init__(self, name="pr", **kwargs):
        super().__init__(name=name, average='micro', mdmc_average='samplewise')

    def compute(self):
        tp, fp, _, fn = self._get_final_stats()
        iou_result = self.iou_compute(tp, fp, fn)
        p5, p6, p7, p8, p9 = self.precision_compute(iou_result)
        return {"P@5": p5, "P@6": p6, "P@7": p7, "P@8": p8, "P@9": p9}

    @no_type_check
    def iou_compute(self, tp, fp, fn) -> Tensor:
        """Compute metric."""
        return _iou_compute(tp, fp, fn, 'none', 'none', self.zero_division)

    def precision_compute(self, iou):
        p5 = torch.sum(iou > 0.5) / iou.size(0)
        p6 = torch.sum(iou > 0.6) / iou.size(0)
        p7 = torch.sum(iou > 0.7) / iou.size(0)
        p8 = torch.sum(iou > 0.8) / iou.size(0)
        p9 = torch.sum(iou > 0.9) / iou.size(0)
        return p5.item(), p6.item(), p7.item(), p8.item(), p9.item()

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class MemMeter:
    """Computes and stores the current, avg, and max of peak Mem usage per iteration"""

    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0  # Per iteration max usage
        self.avg = 0  # Avg per iteration max usage
        self.peak = 0  # Peak usage for lifetime of program
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, n=1, reset_peak_usage=True):
        self.val = torch.cuda.max_memory_allocated() // 1e9
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        self.peak = max(self.peak, self.val)
        if reset_peak_usage:
            torch.cuda.reset_peak_memory_stats()

    def __str__(self):
        fmtstr = (
            "{name}: {val"
            + self.fmt
            + "} ({avg"
            + self.fmt
            + "}/{peak"
            + self.fmt
            + "})"
        )
        return fmtstr.format(**self.__dict__)


def human_readable_time(time_seconds):
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"


class DurationMeter:
    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def add(self, val):
        self.val += val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    def __init__(self, num_batches, meters, real_meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch, enable_print=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [
            " | ".join(
                [
                    f"{os.path.join(name, subname)}: {val:.4f}"
                    for subname, val in meter.compute().items()
                ]
            )
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))
        if enable_print:
            print(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_resume_checkpoint(checkpoint_save_dir):
    if not g_pathmgr.isdir(checkpoint_save_dir):
        return None
    ckpt_file = os.path.join(checkpoint_save_dir, "checkpoint.pt")
    if not g_pathmgr.isfile(ckpt_file):
        return None

    return ckpt_file
