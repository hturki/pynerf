from dataclasses import field, dataclass
from typing import Type, Tuple, List, Dict, Optional

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss, scale_gradients_by_distance_squared
from nerfstudio.model_components.ray_samplers import UniformSampler, ProposalNetworkSampler
from nerfstudio.utils import colormaps
from torch.nn import Parameter

from pynerf.models.pynerf_base_model import PyNeRFBaseModelConfig, PyNeRFBaseModel
from pynerf.pynerf_constants import EXPLICIT_LEVEL, PyNeRFFieldHeadNames, LEVEL_COUNTS, LEVELS


@dataclass
class PyNeRFModelConfig(PyNeRFBaseModelConfig):
    _target: Type = field(default_factory=lambda: PyNeRFModel)

    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    use_gradient_scaling: bool = False
    """Use gradient scaler where the gradients are lower for points closer to the camera."""

class PyNeRFModel(PyNeRFBaseModel):
    config: PyNeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()

        proposal_net_args_list = []
        levels = [6, 8]
        max_res = [512, 2048]
        for i in range(num_prop_nets):
            proposal_net_args_list.append({
                'hidden_dim': 16,
                'log2_hashmap_size': 19,
                'num_levels': levels[i],
                'base_res': self.config.base_resolution,
                'max_res': max_res[i],
                'use_linear': False
            })

        if self.config.use_same_proposal_network:
            assert len(proposal_net_args_list) == 1, 'Only one proposal network is allowed.'
            prop_net_args = proposal_net_args_list[0]
            network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=self.scene_contraction,
                                          **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = proposal_net_args_list[min(i, len(proposal_net_args_list) - 1)]
                network = HashMLPDensityField(self.scene_box.aabb, spatial_distortion=self.scene_contraction,
                                              **prop_net_args)
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.config.proposal_update_every,
        )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.scene_contraction is None:
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups['proposal_networks'] = list(self.proposal_networks.parameters())
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs_inner(self, ray_bundle: RayBundle, explicit_level: Optional[int] = None):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        if explicit_level is not None:
            if ray_samples.metadata is None:
                ray_samples.metadata = {}
            ray_samples.metadata[EXPLICIT_LEVEL] = explicit_level

        field_outputs = self.field(ray_samples)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        outputs = {
            'rgb': self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights),
            'depth': self.renderer_depth(weights=weights, ray_samples=ray_samples)
        }

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        if self.training:
            outputs['weights_list'] = weights_list
            outputs['ray_samples_list'] = ray_samples_list

        if explicit_level is None:
            for i in range(self.config.num_proposal_iterations):
                outputs[f'prop_depth_{i}'] = self.renderer_depth(weights=weights_list[i],
                                                                 ray_samples=ray_samples_list[i])

            outputs['accumulation'] = self.renderer_accumulation(weights=weights)

            if self.training:
                outputs[LEVEL_COUNTS] = field_outputs[PyNeRFFieldHeadNames.LEVEL_COUNTS]
            else:
                levels = field_outputs[PyNeRFFieldHeadNames.LEVELS]
                outputs[LEVELS] = self.renderer_level(weights=weights,
                                                        semantics=levels.clamp(0, self.field.num_scales - 1))

        return outputs

    def get_metrics_dict(self, outputs: Dict[str, any], batch: Dict[str, any]) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)

        if self.training:
            metrics_dict['distortion'] = distortion_loss(outputs['weights_list'], outputs['ray_samples_list'])
            metrics_dict['interlevel'] = interlevel_loss(outputs['weights_list'], outputs['ray_samples_list'])

        return metrics_dict

    def get_loss_dict_inner(self, outputs: Dict[str, any], batch: Dict[str, any],
                            metrics_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict_inner(outputs, batch, metrics_dict)

        if self.training:
            loss_dict['interlevel_loss'] = self.config.interlevel_loss_mult * metrics_dict['interlevel']
            loss_dict['distortion_loss'] = self.config.distortion_loss_mult * metrics_dict['distortion']

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        for i in range(self.config.num_proposal_iterations):
            key = f'prop_depth_{i}'
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs['accumulation'],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
