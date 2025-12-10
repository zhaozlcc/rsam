# _*_ coding: utf-8 _*_#
# Author: zhaozl
# Email : whuzhaozilong@whu.edu.cn
# Licensed under the GNU General Public License v3.0.

import os
import numpy as np
from training.dataset.vos_segment_loader import MySegmentLoader
from training.dataset.vos_raw_dataset import SA1BRawDataset, VOSFrame, VOSVideo
from training.utils.train_utils import get_machine_local_and_dist_rank


class RISORSDataset(SA1BRawDataset):
    def __init__(self, img_folder, gt_folder, file_list_txt=None, excluded_videos_list_txt=None, num_frames=1,
                 mask_area_frac_thresh=1.1, uncertain_iou=-1, negative_to_positive_ratio=1.0, phase="train",
                 seed_value=43):
        super().__init__(img_folder, gt_folder, file_list_txt, excluded_videos_list_txt, num_frames,
                         mask_area_frac_thresh, uncertain_iou)
        self.negative_to_positive_ratio = negative_to_positive_ratio
        _, distributed_rank = get_machine_local_and_dist_rank()
        self.rng = np.random.default_rng(seed=seed_value+distributed_rank)
        self.phase = phase

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        assert idx < len(self.video_names)
        negative_samples = self.rng.choice(self.video_names, size=self.rng.poisson(lam=self.negative_to_positive_ratio))
        video_name = self.video_names[idx]

        video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
        video_mask_path = os.path.join(self.gt_folder, video_name + ".json")
        # negative samples
        negative_mask_paths = []
        for negative_sample in negative_samples:
            if negative_sample != video_name:
                negative_mask_paths.append(os.path.join(self.gt_folder, negative_sample + ".json"))

        segment_loader = MySegmentLoader(
            video_mask_path,
            negative_mask_paths,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        # video id needs to be image_id to be able to load correct annotation file during eval
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader
