from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List

import imageio
import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary
from rich.console import Console

from pynerf.pynerf_constants import WEIGHT, TRAIN_INDEX

CONSOLE = Console(width=120)


def write_mask(dest: Path, w: int, h: int, left_only: bool, right_only: bool) -> None:
    mask = torch.ones(int(h), int(w), dtype=torch.bool)
    if left_only:
        assert not right_only
        mask[:, w // 2:] = False
    if right_only:
        assert not left_only
        mask[:, :w // 2] = False

    tmp_path = dest.parent / f'{uuid.uuid4()}{dest.suffix}'
    Image.fromarray(mask.numpy()).save(tmp_path)
    tmp_path.rename(dest)
    CONSOLE.log(f'Wrote new mask file to {dest}')


@dataclass
class MipNerf360DataParserConfig(DataParserConfig):
    """Mipnerf 360 dataset parser config"""

    _target: Type = field(default_factory=lambda: Mipnerf360)
    """target class to instantiate"""
    data: Path = Path("data/mipnerf360/garden")
    """Directory specifying location of data."""
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    """How much to downscale images."""
    val_skip: int = 8
    """1/val_skip images to use for validation."""
    auto_scale: bool = True
    """Scale based on pose bounds."""
    aabb_scale: float = 1
    """Scene scale."""
    train_split: float = 7 / 8


@dataclass
class Mipnerf360(DataParser):
    """MipNeRF 360 Dataset"""

    config: MipNerf360DataParserConfig

    @classmethod
    def normalize_orientation(cls, poses: np.ndarray):
        """Set the _up_ direction to be in the positive Y direction.
        Args:
            poses: Numpy array of poses.
        """
        poses_orig = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
        center = poses[:, :3, 3].mean(0)
        vec2 = poses[:, :3, 2].sum(0) / np.linalg.norm(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        vec0 = np.cross(up, vec2) / np.linalg.norm(np.cross(up, vec2))
        vec1 = np.cross(vec2, vec0) / np.linalg.norm(np.cross(vec2, vec0))
        c2w = np.stack([vec0, vec1, vec2, center], -1)  # [3, 4]
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)  # [4, 4]
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  # [BS, 1, 4]
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)  # [BS, 4, 4]
        poses = np.linalg.inv(c2w) @ poses
        poses_orig[:, :3, :4] = poses[:, :3, :4]
        return poses_orig

    def _generate_dataparser_outputs(self, split='train'):
        fx = []
        fy = []
        cx = []
        cy = []
        c2ws = []
        width = []
        height = []
        weights = []
        img_scales = []
        image_filenames = []

        camera_params = read_cameras_binary(self.config.data / 'sparse/0/cameras.bin')
        assert camera_params[1].model == 'PINHOLE'
        camera_fx, camera_fy, camera_cx, camera_cy = camera_params[1].params

        for scale in self.config.scales:
            image_dir = "images"
            if scale > 1:
                image_dir += f"_{scale}"
            image_dir = self.config.data / image_dir
            if not image_dir.exists():
                raise ValueError(f"Image directory {image_dir} doesn't exist")

            valid_formats = ['.jpg', '.png']
            num_images = 0
            for f in sorted(image_dir.iterdir()):
                ext = f.suffix
                if ext.lower() not in valid_formats:
                    continue
                image_filenames.append(f)
                num_images += 1

            poses_data = np.load(self.config.data / 'poses_bounds.npy')
            poses = poses_data[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
            bounds = poses_data[:, -2:].transpose([1, 0])

            if num_images != poses.shape[0]:
                raise RuntimeError(f'Different number of images ({num_images}), and poses ({poses.shape[0]})')

            img_0 = imageio.imread(image_filenames[-1])
            image_height, image_width = img_0.shape[:2]

            width.append(torch.full((num_images, 1), image_width, dtype=torch.long))
            height.append(torch.full((num_images, 1), image_height, dtype=torch.long))
            fx.append(torch.full((num_images, 1), camera_fx / scale))
            fy.append(torch.full((num_images, 1), camera_fy / scale))
            cx.append(torch.full((num_images, 1), camera_cx / scale))
            cy.append(torch.full((num_images, 1), camera_cy / scale))
            weights.append(torch.full((num_images,), scale ** 2))
            img_scales.append(torch.full((num_images,), scale, dtype=torch.long))

            # Reorder pose to match nerfstudio convention
            poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=-1)

            # Center poses and rotate. (Compute up from average of all poses)
            poses = self.normalize_orientation(poses)

            # Scale factor used in mipnerf
            if self.config.auto_scale:
                scale_factor = 1 / (np.min(bounds) * 0.75)
                poses[:, :3, 3] *= scale_factor
                bounds *= scale_factor

            # Center poses
            poses[:, :3, 3] = poses[:, :3, 3] - np.mean(poses[:, :3, :], axis=0)[:, 3]
            c2ws.append(torch.from_numpy(poses[:, :3, :4]))

        c2ws = torch.cat(c2ws)
        min_bounds = c2ws[:, :, 3].min(dim=0)[0]
        max_bounds = c2ws[:, :, 3].max(dim=0)[0]

        origin = (max_bounds + min_bounds) * 0.5
        CONSOLE.log('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))

        pose_scale_factor = ((max_bounds - min_bounds) * 0.5).norm().item()
        CONSOLE.log('Calculated pose scale factor: {}'.format(pose_scale_factor))

        for c2w in c2ws:
            c2w[:, 3] = (c2w[:, 3] - origin) / pose_scale_factor
            assert torch.logical_and(c2w >= -1, c2w <= 1).all(), c2w

        # in x,y,z order
        scene_box = SceneBox(aabb=((torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor).float())

        train_indices = set()
        base_train_indices = np.linspace(0, num_images, int(num_images * self.config.train_split),
                                         endpoint=False, dtype=np.int32)
        for i in range(len(self.config.scales)):
            train_indices.update(base_train_indices + num_images * i)

        img_scales = torch.cat(img_scales)
        width = torch.cat(width)
        height = torch.cat(height)

        if split.casefold() == 'train':
            mask_filenames = []
            for i in range(len(image_filenames)):
                if i in train_indices:
                    mask_filenames.append(self.config.data / f'image_full-{img_scales[i].item()}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=False)

                else:
                    mask_filenames.append(self.config.data / f'image_left-{img_scales[i].item()}.png')
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
                    mask_filenames.append(self.config.data / f'image_right-{img_scales[i].item()}.png')
                    if not mask_filenames[-1].exists():
                        write_mask(mask_filenames[-1], w=width[i], h=height[i], left_only=False, right_only=True)

            indices = torch.LongTensor(val_indices)

        cameras = Cameras(
            camera_to_worlds=c2ws[indices].float(),
            fx=torch.cat(fx)[indices],
            fy=torch.cat(fy)[indices],
            cx=torch.cat(cx)[indices],
            cy=torch.cat(cy)[indices],
            width=width[indices],
            height=height[indices],
            camera_type=CameraType.PERSPECTIVE,
        )

        CONSOLE.log('Num images in split {}: {}'.format(split, len(indices)))

        embedding_indices = torch.arange(num_images).unsqueeze(0).repeat(len(self.config.scales), 1).view(-1)

        metadata = {
            TRAIN_INDEX: embedding_indices[indices],
            WEIGHT: torch.cat(weights)[indices],
            'pose_scale_factor': pose_scale_factor,
            'cameras': cameras,
            'near': bounds.min() / pose_scale_factor,
            'far': 10 * bounds.max() / pose_scale_factor
        }

        dataparser_outputs = DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=scene_box,
            dataparser_scale=1,
            mask_filenames=mask_filenames,
            metadata=metadata
        )

        return dataparser_outputs
