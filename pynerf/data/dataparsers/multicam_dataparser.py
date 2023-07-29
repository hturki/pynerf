from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

from pynerf.pynerf_constants import WEIGHT, DEPTH, POSE_SCALE_FACTOR


@dataclass
class MulticamDataParserConfig(DataParserConfig):
    _target: Type = field(default_factory=lambda: Multicam)
    """target class to instantiate"""
    data: Path = Path('data/multicam/lego')
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: Optional[str] = 'white'
    """alpha color of background"""


@dataclass
class Multicam(DataParser):
    config: MulticamDataParserConfig

    def __init__(self, config: MulticamDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split='train'):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        base_meta = load_from_json(self.data / 'metadata.json')
        meta = base_meta[split]
        image_filenames = []
        poses = []
        width = []
        height = []
        focal_length = []
        cx = []
        cy = []
        weights = []
        depth_images = []

        for i in range(len(meta['file_path'])):
            image_filenames.append(self.data / meta['file_path'][i])
            poses.append(np.array(meta['cam2world'][i])[:3])
            width.append(meta['width'][i])
            height.append(meta['height'][i])
            focal_length.append(meta['focal'][i])
            cx.append(meta['width'][i] / 2.0)
            cy.append(meta['height'][i] / 2.0)
            weights.append(meta['lossmult'][i])
            if 'depth_path' in meta:
                depth_images.append(self.data / meta['depth_path'][i])

        poses = np.array(poses).astype(np.float32)
        camera_to_world = torch.from_numpy(poses[:, :3])

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        if 'scene_bounds' in base_meta:
            bounds = torch.FloatTensor(base_meta['scene_bounds'])
        else:
            radius = 1.3 if "ship" not in str(self.data) else 1.5
            bounds = torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]], dtype=torch.float32)

        scene_box = SceneBox(aabb=bounds)

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=torch.FloatTensor(focal_length),
            fy=torch.FloatTensor(focal_length),
            cx=torch.FloatTensor(cx),
            cy=torch.FloatTensor(cy),
            width=torch.IntTensor(width),
            height=torch.IntTensor(height),
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {WEIGHT: weights, 'cameras': cameras, 'near': meta['near'][0], 'far': meta['far'][0]}
        if len(depth_images) > 0:
            metadata[DEPTH] = depth_images
            metadata[POSE_SCALE_FACTOR] = base_meta[POSE_SCALE_FACTOR]

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata
        )

        return dataparser_outputs
