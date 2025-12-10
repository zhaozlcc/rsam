# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import transformers

from PIL import Image as PILImage
from tensordict import tensorclass
from torch.nn.utils.rnn import pad_sequence


@tensorclass
class BatchedVideoMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class DescriptionsData:
    input_ids: torch.LongTensor
    token_type_ids: torch.LongTensor
    attention_mask: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    descriptions: DescriptionsData
    metadata: BatchedVideoMetaData
    categories: list
    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    # Id of the object in the media
    object_id: int
    # Index of the frame in the media (0 if single image)
    frame_index: int
    segment: Union[torch.Tensor, dict]  # RLE dict or binary mask
    description: str
    category: Union[str, int] = 0


@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """Refers to an image/video and all its annotations"""

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key,
    tokenizer: transformers.PreTrainedTokenizer,
    max_tokens: int,
    max_num_objects_total: int = None,
) -> BatchedVideoDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]
    # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_descriptions = [[] for _ in range(T)]
    step_t_category = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [[] for _ in range(T)]  # List to store frame indices for each time step

    samples = {}
    total_objs = 0
    obj_counts = {}
    for video_idx, video in enumerate(batch):
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            total_objs += len(objects)
            samples[(t, video_idx)] = [obj_idx for obj_idx in range(len(objects))]
            obj_counts[(t, video_idx)] = len(objects)

    sampled = {}
    to_remove_count = max(0, total_objs - max_num_objects_total) if max_num_objects_total else 0
    if to_remove_count:
        to_sample = []
        for k, v in samples.items():
            if len(v) > 1:
                idx = random.randrange(0, len(v))
                obj_idx = v.pop(idx)
                sampled[k] = [obj_idx]
                for obj_idx in v:
                    to_sample.append((k, obj_idx))
            else:
                sampled[k] = v
        sampled_ = random.sample(to_sample, len(to_sample)-to_remove_count)
        for v in sampled_:
            sampled[v[0]].append(v[1])
    else:
        sampled = samples

    for k, v in sampled.items():
        video_idx = k[1]
        video = batch[video_idx]
        orig_video_id = video.video_id
        orig_frame_size = video.size
        t = k[0]
        frame = video.frames[t]
        objects = frame.objects
        for obj_idx in v:
            obj = objects[obj_idx]
            orig_obj_id = obj.object_id
            orig_frame_idx = obj.frame_index
            step_t_obj_to_frame_idx[t].append(
                torch.tensor([t, video_idx], dtype=torch.int)
            )
            step_t_masks[t].append(obj.segment.to(torch.bool))
            step_t_descriptions[t].append(obj.description)
            step_t_category[t].append(obj.category)
            step_t_objects_identifier[t].append(
                torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
            )
            step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))

    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    descriptions = [tokenizer(
        sublist,
        padding='longest',
        max_length=max_tokens,
        return_tensors="pt",
        add_special_tokens=True
    ) for sublist in step_t_descriptions]

    categories = []
    for t in range(T):
        categories.append(step_t_category[t])
    input_ids = pad_sequence([d.input_ids for d in descriptions], padding_value=0, batch_first=True)
    token_type_ids = pad_sequence([d.token_type_ids for d in descriptions], padding_value=0, batch_first=True)
    attention_mask = pad_sequence([d.attention_mask for d in descriptions], padding_value=0, batch_first=True)

    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )
    return BatchedVideoDatapoint(
        img_batch=img_batch.contiguous(),
        obj_to_frame_idx=obj_to_frame_idx.contiguous(),
        masks=masks.contiguous(),
        descriptions=DescriptionsData(
            input_ids=input_ids.contiguous(),
            token_type_ids=token_type_ids.contiguous(),
            attention_mask=attention_mask.contiguous(),
        ),
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=objects_identifier.contiguous(),
            frame_orig_size=frame_orig_size.contiguous(),
        ),
        categories=categories,
        dict_key=dict_key,
        batch_size=[T],
    )
