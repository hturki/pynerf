import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageMetadata:
    def __init__(self, image_path: str, W: int, H: int, depth_path: Optional[str], mask_path: Optional[str],
                 weight: Optional[float], train_index: Optional[int], pose_scale_factor: float,
                 local_cache: Optional[Path]):
        self.image_path = image_path
        self.W = W
        self.H = H
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.weight = weight
        self.train_index = train_index
        self._pose_scale_factor = pose_scale_factor
        self._local_cache = local_cache

    def load_image(self) -> torch.Tensor:
        if self._local_cache is not None and not self.image_path.startswith(str(self._local_cache)):
            self.image_path = self._load_from_cache(self.image_path)

        rgbs = Image.open(self.image_path)
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        rgbs = torch.ByteTensor(np.asarray(rgbs))

        if rgbs.shape[-1] == 4:
            rgbs = rgbs.float()
            alpha = rgbs[:, :, -1:] / 255.0
            rgbs = (rgbs[:, :, :3] * alpha + 255 * (1.0 - alpha)).byte()
            # Image.fromarray((rgbs[:, :, :3] * alpha + 255 * (1.0 - alpha)).byte().numpy()).save('/compute/autobot-0-25/hturki/lol2.png')

        return rgbs

    def load_depth(self) -> torch.Tensor:
        if self._local_cache is not None and not self.depth_path.startswith(str(self._local_cache)):
            self.depth_path = self._load_from_cache(self.depth_path)

        depth = torch.FloatTensor(np.load(self.depth_path))

        if depth.shape[0] != self.H or depth.shape[1] != self.W:
            depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(self.H, self.W)).squeeze()

        return depth / self._pose_scale_factor

    def load_mask(self) -> torch.Tensor:
        if self.mask_path is None:
            return torch.ones(self.H, self.W, dtype=torch.bool)

        if self._local_cache is not None and not self.mask_path.startswith(str(self._local_cache)):
            self.mask_path = self._load_from_cache(self.mask_path)

        mask = Image.open(self.mask_path)
        size = mask.size

        if size[0] != self.W or size[1] != self.H:
            mask = mask.resize((self.W, self.H), Image.NEAREST)

        return torch.BoolTensor(np.asarray(mask))

    def _load_from_cache(self, remote_path: str) -> str:
        sha_hash = hashlib.sha256()
        sha_hash.update(remote_path.encode('utf-8'))
        hashed = sha_hash.hexdigest()
        cache_path = self._local_cache / hashed[:2] / hashed[2:4] / f'{hashed}{Path(remote_path).suffix}'

        if cache_path.exists():
            return str(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f'{cache_path}.{uuid.uuid4()}'
        shutil.copy(remote_path, tmp_path)

        os.rename(tmp_path, cache_path)
        return str(cache_path)
