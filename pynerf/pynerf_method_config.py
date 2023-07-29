"""
PyNeRF data configuration file.
"""
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig, ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from pynerf.data.datamanagers.random_subset_datamanager import RandomSubsetDataManagerConfig
from pynerf.data.datamanagers.weighted_datamanager import WeightedDataManagerConfig
from pynerf.data.dataparsers.mipnerf_dataparser import MipNerf360DataParserConfig
from pynerf.data.dataparsers.multicam_dataparser import MulticamDataParserConfig
from pynerf.data.pynerf_pipelines import PyNeRFPipelineConfig, PyNeRFDynamicBatchPipelineConfig
from pynerf.models.pynerf_model import PyNeRFModelConfig
from pynerf.models.pynerf_occupancy_model import PyNeRFOccupancyModelConfig

pynerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name='pynerf',
        steps_per_eval_image=5000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=PyNeRFPipelineConfig(
            datamanager=RandomSubsetDataManagerConfig(
                dataparser=MipNerf360DataParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
            ),
            model=PyNeRFModelConfig(
                eval_num_rays_per_chunk=4096,
            ),
        ),
        optimizers={
            'mlps': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            'proposal_networks': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis='viewer',
        steps_per_eval_all_images=30000
    ),
    description='PyNeRF with proposal network. The default parameters are suited for outdoor scenes.',
)

pynerf_synthetic_method = MethodSpecification(
    config=TrainerConfig(
        method_name='pynerf-synthetic',
        steps_per_eval_all_images=50000,
        max_num_iterations=50001,
        mixed_precision=True,
        pipeline=PyNeRFPipelineConfig(
            datamanager=WeightedDataManagerConfig(
                dataparser=MulticamDataParserConfig(),
                train_num_rays_per_batch=8192),
            model=PyNeRFModelConfig(
                eval_num_rays_per_chunk=8192,
                appearance_embedding_dim=0,
                disable_scene_contraction=True,
                num_nerf_samples_per_ray=96,
                use_gradient_scaling=True,
                max_resolution=65536
            ),
        ),
        optimizers={
            'mlps': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            'proposal_networks': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-15, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis='viewer',
    ),
    description='PyNeRF with proposal network. The default parameters are suited for synthetic scenes.',
)

pynerf_occupancy_method = MethodSpecification(
    config=TrainerConfig(
        method_name='pynerf-occupancy-grid',
        steps_per_eval_all_images=20000,
        max_num_iterations=20001,
        mixed_precision=True,
        pipeline=PyNeRFDynamicBatchPipelineConfig(
            datamanager=WeightedDataManagerConfig(
                dataparser=MulticamDataParserConfig(),
                train_num_rays_per_batch=8192),
            model=PyNeRFOccupancyModelConfig(
                max_resolution=1024,
                eval_num_rays_per_chunk=8192,
                cone_angle=0,
                alpha_thre=0,
                grid_levels=1,
                appearance_embedding_dim=0,
                disable_scene_contraction=True,
                background_color='white',
                output_interpolation='color',
            ),
        ),
        optimizers={
            'mlps': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-8),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=20000),
            },
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-8),
                'scheduler': CosineDecaySchedulerConfig(warm_up_end=512, max_steps=20000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis='viewer',
    ),
    description='PyNeRF with occupancy grid. The default parameters are suited for synthetic scenes.',
)
