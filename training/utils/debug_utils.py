import functools

import numpy as np
from matplotlib import pyplot as plt
from training.utils.data_utils import VideoDatapoint


def show_datapoint(datapoint: VideoDatapoint):
    for frame in datapoint.frames:
        image = frame.data
        for object in frame.objects:
            mask = object.segment
            description = object.description
            object_id = object.object_id

            if image.is_cuda:
                image = image.cpu()
            if mask.is_cuda:
                mask = mask.cpu()
            # 对于带有梯度信息的 tensor，使用 detach() 分离计算图
            image_np = image.detach().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
            mask_np = mask.detach().numpy()
            # 创建两个子图，依次显示 image 和 mask
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
            fig.text(0.5, 0.95, f"Description: {description}, Object ID: {object_id}.", ha='center', va='bottom', fontsize=16)
            # 显示 image（RGB 图像）
            ax1.imshow(image_np)
            ax1.axis('off')
            # 显示 mask，使用灰度 colormap
            ax2.imshow(mask_np, cmap="gray")
            ax2.axis('off')
            plt.tight_layout(rect=(0, 0, 1, 0.9))
            plt.show()


import functools
import torch
import torch.nn.functional as F

def check_scaled_dot_product_attention_inputs(func):
    @functools.wraps(func)
    def wrapper(query, key, value, *args, **kwargs):
        # 打印输入tensor形状
        print(f"[check] query shape: {query.shape}")
        print(f"[check] key shape: {key.shape}")
        print(f"[check] value shape: {value.shape}")

        # 基础形状检查（简单示例）
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            print("[error] query/key/value 必须都是4维张量 (batch, heads, seq_len, dim)")
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            print("[error] batch size 必须匹配")
        if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
            print("[error] num_heads 必须匹配")
        if query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
            print("[error] head dim 必须匹配")

        # 你也可以增加更多检查 ...
        try:
            result = func(query, key, value, *args, **kwargs)
        except RuntimeError as e:
            print(f"[RuntimeError] 在scaled_dot_product_attention调用中捕获: {e}")
            # 这里可以调试时打印更多信息，或者raise再次抛出
            raise
        return result
    return wrapper
