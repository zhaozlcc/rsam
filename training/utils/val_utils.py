import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PIL import Image


def save_result(outputs, targets, unique_objects_identifiers, output_dir=None):
    if output_dir is None:
        output_dir = "results_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert len(outputs) == len(targets)
    for out, target, unique_objects_identifier in zip(outputs, targets, unique_objects_identifiers):
        target_masks = target.float()
        src_masks = out["multistep_pred_masks_high_res"]
        ious = out["multistep_pred_ious"][0]
        object_score_logits = out["multistep_object_score_logits"][0]
        target_masks = target_masks.detach().cpu().numpy()
        src_masks = src_masks.detach().cpu().squeeze().numpy()
        ious = ious.float().detach().cpu().squeeze().numpy()
        object_score_logits = object_score_logits.float().detach().cpu().squeeze().numpy()
        for src_mask, iou, object_score_logit, target_mask, object_id in zip(src_masks, ious, object_score_logits, target_masks, unique_objects_identifier):
            src_mask = src_mask > 0
            result = np.zeros((src_mask.shape[0], src_mask.shape[1], 3))
            # 标记正确区域：白色
            result[np.logical_and(src_mask == target_mask, target_mask == 1)] = [255, 255, 255]  # 正确且为目标：白色
            result[np.logical_and(src_mask == target_mask, target_mask == 0)] = [0, 0, 0]  # 正确且为背景：黑色
            # 标记漏检区域：红色
            result[np.logical_and(src_mask == 0, target_mask == 1)] = [255, 0, 0]  # 漏检：红色
            # 标记误报区域：绿色
            result[np.logical_and(src_mask == 1, target_mask == 0)] = [0, 255, 0]  # 误报：绿色
            result = result.astype(np.uint8)
            Image.fromarray(result).save(os.path.join(output_dir, f"{object_id[0]}-{object_id[1]}.png"))
            m = target_mask.astype(np.uint8) * 255
            Image.fromarray(m).save(os.path.join(output_dir, f"{object_id[0]}-{object_id[1]}-m.png"))
            print(os.path.join(output_dir, f"{object_id[0]}-{object_id[1]}.png"))
            pass

def draw_result(outputs, targets, unique_objects_identifiers, output_dir=None):
    if output_dir is None:
        output_dir = "results_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert len(outputs) == len(targets)
    for out, target, unique_objects_identifier in zip(outputs, targets, unique_objects_identifiers):
        target_masks = target.unsqueeze(1).float()
        # src_masks_list = out["multistep_pred_multimasks_high_res"]
        src_masks_list = out["'multistep_pred_masks_high_res'"]
        ious_list = out["multistep_pred_ious"]
        object_score_logits_list = out["multistep_object_score_logits"]
        for src_masks, ious, object_score_logits in zip(src_masks_list, ious_list, object_score_logits_list):
            src_masks = src_masks.detach().cpu().numpy()
            ious = ious.float().detach().cpu().numpy()
            object_score_logits = object_score_logits.float().detach().cpu().numpy()
            for src_mask, iou, object_score_logit, target_mask, object_id in zip(src_masks, ious, object_score_logits, target_masks, unique_objects_identifier):
                target_mask = target_mask.squeeze(0).detach().cpu().numpy()
                fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(25, 25))
                axes.flatten()[0].imshow(target_mask)
                axes.flatten()[0].axis('off')
                for _ax, _mask, _iou in zip(axes.flatten()[1:], src_mask, iou):
                    _mask = _mask > 0
                    result = np.zeros((_mask.shape[0], _mask.shape[1], 3))
                    # 标记正确区域：白色
                    result[np.logical_and(_mask == target_mask, target_mask == 1)] = [1, 1, 1]  # 正确且为目标：白色
                    result[np.logical_and(_mask == target_mask, target_mask == 0)] = [0, 0, 0]  # 正确且为背景：黑色
                    # 标记漏检区域：红色
                    result[np.logical_and(_mask == 0, target_mask == 1)] = [1, 0, 0]  # 漏检：红色
                    # 标记误报区域：绿色
                    result[np.logical_and(_mask == 1, target_mask == 0)] = [0, 1, 0]  # 误报：绿色
                    _ax.imshow(result, interpolation='none')
                    _ax.set_title(f"Iou Score: {_iou:.4f}", fontsize=18)
                    _ax.axis('off')
                plt.savefig(os.path.join(output_dir, f"{object_id[0]}-{object_id[1]}.png"))
                print(os.path.join(output_dir, f"{object_id[0]}-{object_id[1]}.png"))
                plt.close()

                pass

def get_IoUs(meters):
    tp, fp, fn = torch.tensor(meters.tp), torch.tensor(meters.fp), torch.tensor(meters.fn)
    numerator = tp
    denominator = tp + fp + fn
    IoUs = numerator / denominator
    IoUs = IoUs.cpu().numpy()
    return IoUs
