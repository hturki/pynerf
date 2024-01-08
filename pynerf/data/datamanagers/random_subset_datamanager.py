import random
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union, Literal

import torch
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.comms import get_rank, get_world_size
from rich.console import Console
from torch.nn import Parameter
from torch.utils.data import DistributedSampler, DataLoader

from pynerf.data.dataparsers.multicam_dataparser import MulticamDataParserConfig
from pynerf.data.datasets.image_metadata import ImageMetadata
from pynerf.data.datasets.random_subset_dataset import RandomSubsetDataset
from pynerf.data.weighted_fixed_indices_eval_loader import WeightedFixedIndicesEvalDataloader
from pynerf.pynerf_constants import RGB, WEIGHT, TRAIN_INDEX, DEPTH, POSE_SCALE_FACTOR, RAY_INDEX, RENDER_LEVELS

CONSOLE = Console(width=120)


@dataclass
class RandomSubsetDataManagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: RandomSubsetDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = MulticamDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 4096
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 8192
    """Number of rays per batch to use per eval iteration."""
    eval_image_indices: Optional[Tuple[int, ...]] = None
    """Specifies the image indices to use during eval; if None, uses all val images."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(
        optimizer=AdamOptimizerConfig(lr=6e-6, eps=1e-15),
        scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-4, max_steps=125000))
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    items_per_chunk: int = 25600000
    """Number of entries to load into memory at a time"""
    local_cache_path: Optional[str] = "scratch/pynerf-cache"
    """Caches images and metadata in specific path if set."""
    on_demand_threads: int = 16
    """Number of threads to use when reading data"""
    load_all_in_memory: bool = False
    """Load all of the dataset in memory vs sampling from disk"""


class RandomSubsetDataManager(DataManager):
    """Data manager implementation that samples batches of random pixels/rays/metadata in a chunked manner.
    It can handle datasets that are larger than what can be held in memory

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RandomSubsetDataManagerConfig

    train_dataset: InputDataset
    """Used by the viewer and in various checks in the trainer, but is not actually used to sample batches"""

    def __init__(
            self,
            config: RandomSubsetDataManagerConfig,
            device: Union[torch.device, str] = "cpu",
            test_mode: Literal["test", "val", "inference"] = "test",
            world_size: int = 1,
            local_rank: int = 0,
    ):
        self.test_mode = test_mode # Needed for parent class
        super().__init__()

        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data

        dataparser = self.config.dataparser.setup()
        self.includes_time = dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = dataparser.get_dataparser_outputs(split="train")

        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataparser_outputs.cameras.size, device=self.device)
        self.train_ray_generator = RayGenerator(self.train_dataparser_outputs.cameras.to(self.device),
                                                self.train_camera_optimizer)

        fields_to_load = {RGB}
        for additional_field in {DEPTH, WEIGHT, TRAIN_INDEX}:
            if additional_field in self.train_dataparser_outputs.metadata:
                fields_to_load.add(additional_field)

        self.train_batch_dataset = RandomSubsetDataset(
            items=self._get_image_metadata(self.train_dataparser_outputs),
            fields_to_load=fields_to_load,
            on_demand_threads=self.config.on_demand_threads,
            items_per_chunk=self.config.items_per_chunk,
            load_all_in_memory=self.config.load_all_in_memory,
        )

        self.iter_train_image_dataloader = iter([])
        self.train_dataset = InputDataset(self.train_dataparser_outputs)

        self.eval_dataparser_outputs = dataparser.get_dataparser_outputs(split='test')  # test_mode)

        self.eval_dataset = InputDataset(self.eval_dataparser_outputs)
        self.eval_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.eval_dataparser_outputs.cameras.size, device=self.device)
        self.eval_ray_generator = RayGenerator(self.eval_dataparser_outputs.cameras.to(self.device),
                                               self.eval_camera_optimizer)

        self.eval_image_metadata = self._get_image_metadata(self.eval_dataparser_outputs)
        self.eval_batch_dataset = RandomSubsetDataset(
            items=self.eval_image_metadata,
            fields_to_load=fields_to_load,
            on_demand_threads=self.config.on_demand_threads,
            items_per_chunk=(self.config.eval_num_rays_per_batch * 10),
            load_all_in_memory=self.config.load_all_in_memory
        )

        self.iter_eval_batch_dataloader = iter([])

    @cached_property
    def fixed_indices_eval_dataloader(self):
        image_indices = []
        for item_index in range(get_rank(), len(self.eval_dataparser_outputs.cameras), get_world_size()):
            image_indices.append(item_index)

        return WeightedFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
            image_indices=image_indices
        )

    def _set_train_loader(self):
        batch_size = self.config.train_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.train_sampler = DistributedSampler(self.train_batch_dataset, self.world_size, self.local_rank)
            assert self.config.train_num_rays_per_batch % self.world_size == 0
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size,
                                                     sampler=self.train_sampler, num_workers=0, pin_memory=True)
        else:
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=0, pin_memory=True)

        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def _set_eval_batch_loader(self):
        batch_size = self.config.eval_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.eval_sampler = DistributedSampler(self.eval_batch_dataset, self.world_size, self.local_rank)
            assert self.config.eval_num_rays_per_batch % self.world_size == 0
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size,
                                                    sampler=self.eval_sampler,
                                                    num_workers=0, pin_memory=True)
        else:
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=0, pin_memory=True)

        self.iter_eval_batch_dataloader = iter(self.eval_batch_dataloader)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        batch = next(self.iter_train_image_dataloader, None)
        if batch is None:
            self.train_batch_dataset.load_chunk()
            self._set_train_loader()
            batch = next(self.iter_train_image_dataloader)

        ray_bundle = self.train_ray_generator(batch[RAY_INDEX])
        self.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        batch = next(self.iter_eval_batch_dataloader, None)
        if batch is None:
            self.eval_batch_dataset.load_chunk()
            self._set_eval_batch_loader()
            batch = next(self.iter_eval_batch_dataloader)

        ray_bundle = self.eval_ray_generator(batch[RAY_INDEX])
        self.transfer_train_index(ray_bundle, batch)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_index = random.choice(self.fixed_indices_eval_dataloader.image_indices)
        ray_bundle, batch = self.fixed_indices_eval_dataloader.get_data_from_image_idx(image_index)

        metadata = self.eval_image_metadata[image_index]

        if ray_bundle.metadata is None:
            ray_bundle.metadata = {}
            ray_bundle.metadata[RENDER_LEVELS] = True

        if metadata.train_index is not None:
            ray_bundle.metadata[TRAIN_INDEX] = torch.full_like(ray_bundle.camera_indices, metadata.train_index,
                                                               dtype=torch.int64)
        if metadata.weight is not None:
            batch[WEIGHT] = torch.full_like(ray_bundle.camera_indices, metadata.weight, dtype=torch.float32)

        if metadata.depth_path is not None:
            batch[DEPTH] = metadata.load_depth().to(ray_bundle.camera_indices.device).unsqueeze(-1)

        return image_index, ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups

    def _get_image_metadata(self, outputs: DataparserOutputs) -> List[ImageMetadata]:
        local_cache_path = Path(self.config.local_cache_path) if self.config.local_cache_path is not None else None

        items = []
        for i in range(len(outputs.image_filenames)):
            items.append(
                ImageMetadata(str(outputs.image_filenames[i]),
                              int(outputs.cameras.width[i]),
                              int(outputs.cameras.height[i]),
                              outputs.metadata[DEPTH][i] if DEPTH in outputs.metadata else None,
                              str(outputs.mask_filenames[i]) if outputs.mask_filenames is not None else None,
                              float(outputs.metadata[WEIGHT][i]) if WEIGHT in outputs.metadata else None,
                              int(outputs.metadata[TRAIN_INDEX][i]) if TRAIN_INDEX in outputs.metadata else None,
                              outputs.metadata[POSE_SCALE_FACTOR] if POSE_SCALE_FACTOR in outputs.metadata else 1,
                              local_cache_path))

        return items

    @staticmethod
    def transfer_train_index(ray_bundle: RAY_INDEX, batch: Dict) -> None:
        if TRAIN_INDEX in batch:
            if ray_bundle.metadata is None:
                ray_bundle.metadata = {}
            ray_bundle.metadata[TRAIN_INDEX] = batch[TRAIN_INDEX].unsqueeze(-1).to(ray_bundle.origins.device)
            del batch[TRAIN_INDEX]
