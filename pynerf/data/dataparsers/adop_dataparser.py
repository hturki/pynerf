from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from pyquaternion import Quaternion

from pynerf.data.dataparsers.mipnerf_dataparser import write_mask
from pynerf.pynerf_constants import TRAIN_INDEX, WEIGHT

OPENCV_TO_OPENGL = torch.DoubleTensor([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]])

DOWN_TO_FORWARD = torch.DoubleTensor([[1, 0, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, 0, 1]])


@dataclass
class AdopDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: Adop)
    """target class to instantiate"""
    data: Path = Path("data/adop/boat")

    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""

    train_split: float = 0.9

    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass
class Adop(DataParser):
    config: AdopDataParserConfig

    def __init__(self, config: AdopDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor

    def get_dataparser_outputs(self, split="train", scales: Optional[List[int]] = None):
        with (self.config.data / "images.txt").open() as f:
            image_paths = f.readlines()

        with (self.config.data / "adop-poses.txt").open() as f:
            poses = f.readlines()

        with (self.config.data / "undistorted_intrinsics_adop.txt").open() as f:
            intrinsics = f.readlines()

        num_images_base = len(image_paths)
        assert num_images_base == len(poses) == len(intrinsics)

        image_filenames = []
        c2ws = []
        width = []
        height = []
        fx = []
        fy = []
        cx = []
        cy = []
        weights = []
        img_scales = []

        near = 1e10
        far = -1

        if scales is None:
            scales = self.config.scales
        for scale in scales:
            for image_path, c2w_line, K in zip(image_paths, poses, intrinsics):
                image_filenames.append(self.data / f'undistorted_images_adop-{scale}' / image_path.strip())

                pose_line = [float(x) for x in c2w_line.strip().split()]
                w2c = torch.DoubleTensor(
                    Quaternion(w=pose_line[3], x=pose_line[0], y=pose_line[1], z=pose_line[2]).transformation_matrix)
                w2c[:3, 3] = torch.DoubleTensor(pose_line[4:7])

                # Some points seem to be extremely close
                # if near > 0.1:
                near = min(pose_line[7], near)
                far = max(pose_line[8], far)
                c2w = torch.inverse(w2c)

                c2ws.append((DOWN_TO_FORWARD @ (c2w @ OPENCV_TO_OPENGL))[:3].unsqueeze(0))

                K_line = [float(x) for x in K.strip().split()]
                width.append(int(K_line[0]) // scale)
                height.append(int(K_line[1]) // scale)
                fx.append(K_line[2] / scale)
                fy.append(K_line[6] / scale)
                cx.append(K_line[4] / scale)
                cy.append(K_line[7] / scale)
                weights.append(scale ** 2)
                img_scales.append(scale)

        c2ws = torch.cat(c2ws)
        min_bounds = c2ws[:, :, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :, 3].max(dim=0)[0]

        origin = (max_bounds + min_bounds) * 0.5
        print('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))

        pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
        print('Calculated pose scale factor: {}'.format(pose_scale_factor))

        for c2w in c2ws:
            c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
            assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w

        # in x,y,z order
        c2ws[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=((torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor).float())

        train_indices = set()
        base_train_indices = np.linspace(0, num_images_base, int(num_images_base * self.config.train_split),
                                         endpoint=False, dtype=np.int32)
        for i in range(len(scales)):
            train_indices.update(base_train_indices + num_images_base * i)

        if split.casefold() == 'train':
            mask_filenames = []
            for i in range(len(image_filenames)):
                if i in train_indices:
                    mask_filenames.append(self.data / f'image_full-{img_scales[i]}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=False)
                else:
                    mask_filenames.append(self.data / f'image_left-{img_scales[i]}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=True, right_only=False)

            indices = torch.arange(len(image_filenames), dtype=torch.long)
        else:
            val_indices = []
            mask_filenames = []
            train_indices = set(train_indices)
            for i in range(len(image_filenames)):
                if i not in train_indices:
                    val_indices.append(i)
                    mask_filenames.append(self.data / f'image_right-{img_scales[i]}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=True)

            indices = torch.LongTensor(val_indices)

        cameras = Cameras(
            camera_to_worlds=c2ws[indices].float(),
            fx=torch.FloatTensor(fx)[indices],
            fy=torch.FloatTensor(fy)[indices],
            cx=torch.FloatTensor(cx)[indices],
            cy=torch.FloatTensor(cy)[indices],
            width=torch.IntTensor(width)[indices],
            height=torch.IntTensor(height)[indices],
            camera_type=CameraType.PERSPECTIVE,
        )

        print('Num images in split {}: {}'.format(split, len(indices)))

        embedding_indices = torch.arange(num_images_base).unsqueeze(0).repeat(len(scales), 1).view(-1)

        metadata = {
            TRAIN_INDEX: embedding_indices[indices],
            WEIGHT: torch.FloatTensor(weights)[indices],
            "pose_scale_factor": pose_scale_factor,
            "cameras": cameras,
            'near': near / pose_scale_factor,
            'far': far * 10 / pose_scale_factor
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            mask_filenames=mask_filenames,
            metadata=metadata
        )

        return dataparser_outputs
