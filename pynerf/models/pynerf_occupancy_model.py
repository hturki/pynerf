from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import math
import nerfacc
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.utils import colormaps
from rich.console import Console

from pynerf.models.pynerf_base_model import PyNeRFBaseModelConfig, PyNeRFBaseModel
from pynerf.pynerf_constants import EXPLICIT_LEVEL, PyNeRFFieldHeadNames, LEVEL_COUNTS, LEVELS
from pynerf.samplers.pynerf_vol_sampler import PyNeRFVolumetricSampler

CONSOLE = Console(width=120)


@dataclass
class PyNeRFOccupancyModelConfig(PyNeRFBaseModelConfig):
    _target: Type = field(
        default_factory=lambda: PyNeRFOccupancyModel
    )

    max_num_samples_per_ray: int = 1024
    """Number of samples in field evaluation."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""


class PyNeRFOccupancyModel(PyNeRFBaseModel):
    config: PyNeRFOccupancyModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.render_step_size is None:
            self.render_step_size = (
                    (self.scene_box.aabb[1] - self.scene_box.aabb[0]).max()
                    * math.sqrt(3)
                    / self.config.max_num_samples_per_ray
            ).item()
            CONSOLE.log(f'Setting render step size to {self.render_step_size}')
        else:
            self.render_step_size = self.config.render_step_size

        # Occupancy Grid
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_box.aabb.flatten(),
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = PyNeRFVolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        def update_occupancy_grid(step: int):
            self.occupancy_grid.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: self.field.density_fn(x, step_size=self.render_step_size) * self.render_step_size,
            )

        callbacks = [
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=update_occupancy_grid,
            ),
        ]

        return callbacks

    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        num_rays = len(ray_bundle)

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=self.near,
                far_plane=self.far,
                render_step_size=self.render_step_size,
                alpha_thre=self.config.alpha_thre,
                cone_angle=self.config.cone_angle,
            )

        if explicit_level is not None:
            if ray_samples.metadata is None:
                ray_samples.metadata = {}
            ray_samples.metadata[EXPLICIT_LEVEL] = explicit_level

        field_outputs = self.field(ray_samples)

        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)

        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.frustums.starts[..., 0],
            t_ends=ray_samples.frustums.ends[..., 0],
            sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
            packed_info=packed_info,
        )[0][..., None]

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights, ray_indices=ray_indices,
                                num_rays=num_rays, )
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples, ray_indices=ray_indices,
                                    num_rays=num_rays)

        outputs = {'rgb': rgb, 'depth': depth}

        if explicit_level is None:
            accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
            alive_ray_mask = accumulation.squeeze(-1) > 0

            outputs['accumulation'] = accumulation
            outputs['alive_ray_mask'] = alive_ray_mask  # the rays we kept from sampler
            outputs['num_samples_per_ray'] = packed_info[:, 1]

            if self.training:
                outputs[LEVEL_COUNTS] = field_outputs[PyNeRFFieldHeadNames.LEVEL_COUNTS]
            else:
                levels = field_outputs[PyNeRFFieldHeadNames.LEVELS]
                outputs[LEVELS] = self.renderer_level(weights=weights,
                                                      semantics=levels.clamp(0, self.field.num_scales - 1),
                                                      ray_indices=ray_indices, num_rays=num_rays)

        return outputs

    def get_metrics_dict(self, outputs: Dict[str, any], batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        metrics_dict['num_samples_per_batch'] = outputs['num_samples_per_ray'].sum()

        return metrics_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        images_dict['alive_ray_mask'] = colormaps.apply_colormap(outputs['alive_ray_mask'])

        return metrics_dict, images_dict
