"""
PyNeRF field implementation.
"""
from abc import abstractmethod
from collections import defaultdict
from enum import Enum, auto
from typing import Tuple, Optional, Any, Dict, List

import math
import tinycudann as tcnn
import torch
import torch_scatter
from jaxtyping import Shaped
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components import MLP
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from rich.console import Console
from torch import nn, Tensor

from pynerf.pynerf_constants import TRAIN_INDEX, PyNeRFFieldHeadNames, EXPLICIT_LEVEL

CONSOLE = Console(width=120)


class OutputInterpolation(Enum):
    COLOR = auto()
    EMBEDDING = auto()


class LevelInterpolation(Enum):
    NONE = auto()
    LINEAR = auto()


class PyNeRFBaseField(Field):

    def __init__(
            self,
            aabb,
            num_images: int,
            encoding_input_dims: List[int],
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
            trained_level_resolution: Optional[int] = 128,
    ) -> None:
        super().__init__()
        self.register_buffer('aabb', aabb, persistent=False)

        self.geo_feat_dim = geo_feat_dim
        self.appearance_embedding_dim = appearance_embedding_dim
        self.output_interpolation = output_interpolation
        self.level_interpolation = level_interpolation
        self.spatial_distortion = spatial_distortion
        self.num_scales = num_scales
        self.share_feature_grid = share_feature_grid
        self.cameras = cameras

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SphericalHarmonics',
                'degree': 4,
            },
        )

        if appearance_embedding_dim > 0:
            self.embedding_appearance = Embedding(num_images, self.appearance_embedding_dim)

        area_of_interest = (aabb[1] - aabb[0]).max()
        if self.spatial_distortion is not None:
            area_of_interest *= 2  # Scene contraction uses half of the table capacity for contracted space

        self.log_scale_factor = math.log(scale_factor)

        # Get base log of lowest mip level
        self.base_log = math.log(area_of_interest, scale_factor) \
                        - (math.log(max_resolution, scale_factor) - (num_scales - 1))

        self.trained_level_resolution = trained_level_resolution
        if trained_level_resolution is not None:
            self.register_buffer('min_trained_level', torch.full(
                (trained_level_resolution, trained_level_resolution, trained_level_resolution), self.num_scales,
                dtype=torch.float32))
            self.register_buffer('max_trained_level', torch.full(
                (trained_level_resolution, trained_level_resolution, trained_level_resolution), -1,
                dtype=torch.float32))

        mlp_bases = []

        self.encoding_input_dims = encoding_input_dims
        assert len(encoding_input_dims) == num_scales, \
            f'Number of encoding dims {len(encoding_input_dims)} different from number of scales {num_scales}'
        for encoding_input_dim in encoding_input_dims:
            mlp_bases.append(MLP(
                in_dim=encoding_input_dim,
                out_dim=1 + self.geo_feat_dim,
                num_layers=num_layers,
                layer_width=hidden_dim,
                activation=nn.ReLU(),
                out_activation=None,
                implementation="tcnn"
            ))

        self.mlp_bases = nn.ModuleList(mlp_bases)

        if output_interpolation == OutputInterpolation.COLOR:
            mlp_heads = []
            for i in range(num_scales):
                mlp_heads.append(MLP(
                    in_dim=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
                    out_dim=3,
                    num_layers=num_layers_color,
                    layer_width=hidden_dim_color,
                    activation=nn.ReLU(),
                    out_activation=nn.Sigmoid(),
                    implementation="tcnn"
                ))
            self.mlp_heads = nn.ModuleList(mlp_heads)
        else:
            self.mlp_head = MLP(
                in_dim=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
                out_dim=3,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation="tcnn"
            )

    @abstractmethod
    def get_shared_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_level_encoding(self, level: int, positions: torch.Tensor) -> torch.Tensor:
        pass

    def get_density(self, ray_samples: RaySamples, update_levels: bool = True):
        positions = ray_samples.frustums.get_positions()

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
            positions_flat = positions.view(-1, 3)
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
            positions_flat = positions.view(-1, 3)

        explicit_level = ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata
        if explicit_level:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            pixel_levels = torch.full_like(ray_samples.frustums.starts[..., 0], level).view(-1)
        else:
            # Assuming pixels are square
            sample_distances = ((ray_samples.frustums.starts + ray_samples.frustums.ends) / 2)
            pixel_widths = (ray_samples.frustums.pixel_area.sqrt() * sample_distances).view(-1)

            pixel_levels = self.base_log - torch.log(pixel_widths) / self.log_scale_factor
            if self.trained_level_resolution is not None:
                reso_indices = (positions_flat * self.trained_level_resolution).floor().long().clamp(0,
                                                                                                     self.trained_level_resolution - 1)
                if self.training:
                    if update_levels:
                        flat_indices = reso_indices[
                                           ..., 0] * self.trained_level_resolution * self.trained_level_resolution \
                                       + reso_indices[..., 1] * self.trained_level_resolution + reso_indices[..., 2]
                        torch_scatter.scatter_min(pixel_levels, flat_indices, out=self.min_trained_level.view(-1))
                        torch_scatter.scatter_max(pixel_levels, flat_indices, out=self.max_trained_level.view(-1))
                else:
                    min_levels = self.min_trained_level[
                        reso_indices[..., 0], reso_indices[..., 1], reso_indices[..., 2]]
                    max_levels = self.max_trained_level[
                        reso_indices[..., 0], reso_indices[..., 1], reso_indices[..., 2]]
                    pixel_levels = torch.maximum(min_levels, torch.minimum(pixel_levels, max_levels))

        if self.level_interpolation == LevelInterpolation.NONE:
            level_indices = get_levels(pixel_levels, self.num_scales)
            level_weights = {}
            for level, indices in level_indices.items():
                level_weights[level] = torch.ones_like(indices, dtype=pixel_levels.dtype)
        elif self.level_interpolation == LevelInterpolation.LINEAR:
            level_indices, level_weights = get_weights(pixel_levels, self.num_scales)
        else:
            raise Exception(self.level_interpolation)

        if self.share_feature_grid:
            encoding = self.get_shared_encoding(positions_flat)

        if self.output_interpolation == OutputInterpolation.COLOR:
            density = None
            level_embeddings = {}
        else:
            interpolated_h = None

        for level, cur_level_indices in level_indices.items():
            if self.share_feature_grid:
                level_encoding = encoding[cur_level_indices][..., :self.encoding_input_dims[level]]
            else:
                level_encoding = self.get_level_encoding(level, positions_flat[cur_level_indices])

            level_h = self.mlp_bases[level](level_encoding).to(positions)
            cur_level_weights = level_weights[level]
            if self.output_interpolation == OutputInterpolation.COLOR:
                density_before_activation, level_mlp_out = torch.split(level_h, [1, self.geo_feat_dim], dim=-1)
                level_embeddings[level] = level_mlp_out
                level_density = trunc_exp(density_before_activation - 1)
                if density is None:
                    density = torch.zeros(positions_flat.shape[0], *level_density.shape[1:],
                                          dtype=level_density.dtype, device=level_density.device)
                density[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_density
            elif self.output_interpolation == OutputInterpolation.EMBEDDING:
                if interpolated_h is None:
                    interpolated_h = torch.zeros(positions_flat.shape[0], *level_h.shape[1:],
                                                 dtype=level_h.dtype, device=level_h.device)

                interpolated_h[cur_level_indices] += cur_level_weights.unsqueeze(-1) * level_h
            else:
                raise Exception(self.output_interpolation)

        if self.output_interpolation == OutputInterpolation.COLOR:
            additional_info = (level_indices, level_weights, level_embeddings)
        elif self.output_interpolation == OutputInterpolation.EMBEDDING:
            density_before_activation, mlp_out = torch.split(interpolated_h, [1, self.geo_feat_dim], dim=-1)
            density = trunc_exp(density_before_activation - 1)
            additional_info = mlp_out
        else:
            raise Exception(self.output_interpolation)

        if self.training:
            level_counts = defaultdict(int)
            for level, indices in level_indices.items():
                level_counts[level] = indices.shape[0] / pixel_levels.shape[0]

            return density.view(ray_samples.frustums.starts.shape), (additional_info, level_counts)
        else:
            return density.view(ray_samples.frustums.starts.shape), (additional_info, pixel_levels.view(density.shape))

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tuple[Any] = None):
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        if self.appearance_embedding_dim > 0:
            if ray_samples.metadata is not None and TRAIN_INDEX in ray_samples.metadata:
                embedded_appearance = self.embedding_appearance(
                    ray_samples.metadata[TRAIN_INDEX].squeeze().to(d.device))
            elif self.training:
                embedded_appearance = self.embedding_appearance(ray_samples.camera_indices.squeeze())
            else:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)

            embedded_appearance = embedded_appearance.view(-1, self.appearance_embedding_dim)

        outputs = {}

        if self.training:
            density_embedding, level_counts = density_embedding
            outputs[PyNeRFFieldHeadNames.LEVEL_COUNTS] = level_counts
        else:
            density_embedding, levels = density_embedding
            outputs[PyNeRFFieldHeadNames.LEVELS] = levels.view(ray_samples.frustums.starts.shape)

        if ray_samples.metadata is not None and EXPLICIT_LEVEL in ray_samples.metadata:
            level = ray_samples.metadata[EXPLICIT_LEVEL]
            if self.output_interpolation == OutputInterpolation.COLOR:
                _, _, level_embeddings = density_embedding
                density_embedding = level_embeddings[level]

                mlp_head = self.mlp_heads[level]
            else:
                mlp_head = self.mlp_head

            color_inputs = [d, density_embedding]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance)

            h = torch.cat(color_inputs, dim=-1)

            outputs[FieldHeadNames.RGB] = mlp_head(h).view(directions.shape).to(directions)
            return outputs

        if self.output_interpolation != OutputInterpolation.COLOR:
            color_inputs = [d, density_embedding]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance)
            h = torch.cat(color_inputs, dim=-1)
            rgbs = self.mlp_head(h).view(directions.shape).to(directions)
            outputs[FieldHeadNames.RGB] = rgbs

            return outputs

        level_indices, level_weights, level_embeddings = density_embedding

        rgbs = None
        for level, cur_level_indices in level_indices.items():
            color_inputs = [d[cur_level_indices], level_embeddings[level]]
            if self.appearance_embedding_dim > 0:
                color_inputs.append(embedded_appearance[cur_level_indices])
            h = torch.cat(color_inputs, dim=-1)

            level_rgbs = self.mlp_heads[level](h).to(directions)
            if rgbs is None:
                rgbs = torch.zeros_like(directions)
            rgbs.view(-1, 3)[cur_level_indices] += level_weights[level].unsqueeze(-1) * level_rgbs

        outputs[FieldHeadNames.RGB] = rgbs
        return outputs

    def density_fn(self, positions: Shaped[Tensor, "*bs 3"], times: Optional[Shaped[Tensor, "*bs 1"]] = None,
                   step_size: int = None, origins: Optional[Shaped[Tensor, "*bs 3"]] = None,
                   directions: Optional[Shaped[Tensor, "*bs 3"]] = None,
                   starts: Optional[Shaped[Tensor, "*bs 1"]] = None,
                   ends: Optional[Shaped[Tensor, "*bs 1"]] = None, pixel_area: Optional[Shaped[Tensor, "*bs 1"]] = None) \
            -> Shaped[Tensor, "*bs 1"]:
        """Returns only the density. Used primarily with the density grid.
        """
        if origins is None:
            camera_ids = torch.randint(0, len(self.cameras), (positions.shape[0],), device=positions.device)
            cameras = self.cameras.to(camera_ids.device)[camera_ids]
            origins = cameras.camera_to_worlds[:, :, 3]
            directions = positions - origins
            directions, _ = camera_utils.normalize_with_norm(directions, -1)
            coords = torch.cat(
                [torch.rand_like(origins[..., :1]) * cameras.height, torch.rand_like(origins[..., :1]) * cameras.width],
                -1).floor().long()

            pixel_area = cameras.generate_rays(torch.arange(len(cameras)).unsqueeze(-1), coords=coords).pixel_area
            starts = (origins - positions).norm(dim=-1, keepdim=True) - step_size / 2
            ends = starts + step_size

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=directions,
                starts=starts,
                ends=ends,
                pixel_area=pixel_area,
            ),
            times=times
        )

        density, _ = self.get_density(ray_samples, update_levels=False)
        return density


@torch.jit.script
def get_levels(pixel_levels: torch.Tensor, num_levels: int) -> Dict[int, torch.Tensor]:
    sorted_pixel_levels, ordering = pixel_levels.sort(descending=False)
    level_indices: Dict[int, torch.Tensor] = {}

    start = 0
    for level in range(num_levels - 1):
        end = start + (sorted_pixel_levels[start:] <= level).sum()

        if end > start:
            if ordering[start:end].shape[0] > 0:
                level_indices[level] = ordering[start:end]

        start = end

    if ordering[start:].shape[0] > 0:
        level_indices[num_levels - 1] = ordering[start:]

    return level_indices


@torch.jit.script
def get_weights(pixel_levels: torch.Tensor, num_levels: int) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    sorted_pixel_levels, ordering = pixel_levels.sort(descending=False)
    level_indices: Dict[int, torch.Tensor] = {}
    level_weights: Dict[int, torch.Tensor] = {}

    mid = 0
    end = 0
    for level in range(num_levels):
        if level == 0:
            mid = (sorted_pixel_levels < level).sum()
            cur_level_indices = [ordering[:mid]]
            cur_level_weights = [torch.ones_like(cur_level_indices[0], dtype=pixel_levels.dtype)]
        else:
            start = mid
            mid = end
            cur_level_indices = [ordering[start:mid]]
            cur_level_weights = [sorted_pixel_levels[start:mid] - (level - 1)]

        if level < num_levels - 1:
            end = mid + (sorted_pixel_levels[mid:] < level + 1).sum()
            cur_level_indices.append(ordering[mid:end])
            cur_level_weights.append(1 - (sorted_pixel_levels[mid:end] - level))
        else:
            cur_level_indices.append(ordering[mid:])
            cur_level_weights.append(torch.ones_like(cur_level_indices[-1], dtype=pixel_levels.dtype))

        cur_level_indices = torch.cat(cur_level_indices)

        if cur_level_indices.shape[0] > 0:
            level_indices[level] = cur_level_indices
            level_weights[level] = torch.cat(cur_level_weights)

    return level_indices, level_weights


def parse_output_interpolation(model: str) -> OutputInterpolation:
    if model.casefold() == 'color':
        return OutputInterpolation.COLOR
    elif model.casefold() == 'embedding':
        return OutputInterpolation.EMBEDDING
    else:
        raise Exception(model)


def parse_level_interpolation(model: str) -> LevelInterpolation:
    if model.casefold() == 'none':
        return LevelInterpolation.NONE
    elif model.casefold() == 'linear':
        return LevelInterpolation.LINEAR
    else:
        raise Exception(model)
