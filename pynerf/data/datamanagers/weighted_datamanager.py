from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Type, Tuple, Dict,
)

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager
from nerfstudio.data.pixel_samplers import (
    PixelSamplerConfig,
)

from pynerf.data.datamanagers.random_subset_datamanager import RandomSubsetDataManager
from pynerf.data.datasets.weighted_dataset import WeightedDataset
from pynerf.data.weighted_pixel_sampler import WeightedPixelSampler
from pynerf.pynerf_constants import RENDER_LEVELS


@dataclass
class WeightedPixelSamplerConfig(PixelSamplerConfig):
    _target: Type = field(default_factory=lambda: WeightedPixelSampler)

@dataclass
class WeightedDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: WeightedDataManager)

    pixel_sampler: WeightedPixelSamplerConfig = field(default_factory=lambda: WeightedPixelSamplerConfig())


class WeightedDataManager(VanillaDataManager[WeightedDataset]):

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        RandomSubsetDataManager.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        RandomSubsetDataManager.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_index, ray_bundle, batch = super().next_eval_image(step)
        RandomSubsetDataManager.transfer_train_index(ray_bundle, batch)

        if ray_bundle.metadata is None:
            ray_bundle.metadata = {}
            ray_bundle.metadata[RENDER_LEVELS] = True

        return image_index, ray_bundle, batch
