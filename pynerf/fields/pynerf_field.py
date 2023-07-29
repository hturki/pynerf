from typing import Optional

import math
import tinycudann as tcnn
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from torch import nn

from pynerf.fields.pynerf_base_field import PyNeRFBaseField, OutputInterpolation, LevelInterpolation


class PyNeRFField(PyNeRFBaseField):

    def __init__(
            self,
            aabb,
            num_images: int,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            appearance_embedding_dim: int = 32,
            max_resolution: int = 4096,
            spatial_distortion: SpatialDistortion = None,
            output_interpolation: OutputInterpolation = OutputInterpolation.EMBEDDING,
            level_interpolation: LevelInterpolation = LevelInterpolation.LINEAR,
            num_scales: int = 8,
            scale_factor: float = 2.0,
            share_feature_grid: bool = False,
            cameras: Cameras = None,
            base_resolution: int = 16,
            features_per_level: int = 2,
            num_levels: int = 16,
            log2_hashmap_size: int = 19,
            trained_level_resolution: Optional[int] = 128,
    ) -> None:
        super().__init__(aabb, num_images, [num_levels * features_per_level for _ in range(num_scales)], num_layers,
                         hidden_dim, geo_feat_dim, num_layers_color, hidden_dim_color, appearance_embedding_dim,
                         max_resolution, spatial_distortion, output_interpolation, level_interpolation, num_scales,
                         scale_factor, share_feature_grid, cameras, trained_level_resolution)

        if not share_feature_grid:
            encodings = []

            for scale in range(num_scales):
                cur_max_res = max_resolution / (scale_factor ** (num_scales - 1 - scale))
                cur_level_scale = math.exp(math.log(cur_max_res / base_resolution) / (num_levels - 1))

                encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        'otype': 'HashGrid',
                        'n_levels': num_levels,
                        'n_features_per_level': features_per_level,
                        'log2_hashmap_size': log2_hashmap_size,
                        'base_resolution': base_resolution,
                        'per_level_scale': cur_level_scale,
                    })

                encodings.append(encoding)

            self.encodings = nn.ModuleList(encodings)
        else:
            per_level_scale = math.exp(math.log(max_resolution / base_resolution) / (num_levels - 1))

            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': num_levels,
                    'n_features_per_level': features_per_level,
                    'log2_hashmap_size': log2_hashmap_size,
                    'base_resolution': base_resolution,
                    'per_level_scale': per_level_scale,
                })

    def get_shared_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        return self.encoding(positions)

    def get_level_encoding(self, level: int, positions: torch.Tensor) -> torch.Tensor:
        return self.encodings[level](positions)
