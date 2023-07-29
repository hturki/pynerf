from typing import Optional, Callable, Tuple

import torch
from jaxtyping import Float
from nerfacc import OccGridEstimator
from nerfstudio.cameras.rays import RaySamples, RayBundle, Frustums
from nerfstudio.model_components.ray_samplers import Sampler
from torch import Tensor


class PyNeRFVolumetricSampler(Sampler):
    """
    Similar to VolumetricSampler in NerfStudio, but passes additional camera ray information to density_fn
    """

    def __init__(
            self,
            occupancy_grid: OccGridEstimator,
            density_fn: Optional[Callable] = None,
    ):
        super().__init__()
        assert occupancy_grid is not None
        self.density_fn = density_fn
        self.occupancy_grid = occupancy_grid

    def get_sigma_fn(self, origins, directions, pixel_area, times=None) -> Optional[Callable]:
        """Returns a function that returns the density of a point.

        Args:Ø
            origins: Origins of rays
            directions: Directions of rays
            pixel_area: Pixel area of rays
            times: Times at which rays are sampled
        Returns:
            Function that returns the density of a point or None if a density function is not provided.
        """

        if self.density_fn is None or not self.training:
            return None

        density_fn = self.density_fn

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = origins[ray_indices]
            t_dirs = directions[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            return density_fn(positions, times=times[ray_indices] if times is not None else None, origins=t_origins,
                              directions=t_dirs, starts=t_starts[:, None], ends=t_ends[:, None],
                              pixel_area=pixel_area[ray_indices]).squeeze(-1)

        return sigma_fn

    def generate_ray_samples(self) -> RaySamples:
        raise RuntimeError(
            "The VolumetricSampler fuses sample generation and density check together. Please call forward() directly."
        )

    def forward(
            self,
            ray_bundle: RayBundle,
            render_step_size: float,
            near_plane: float = 0.0,
            far_plane: Optional[float] = None,
            alpha_thre: float = 0.01,
            cone_angle: float = 0.0,
    ) -> Tuple[RaySamples, Float[Tensor, "total_samples "]]:
        """Generate ray samples in a bounding box.

        Args:
            ray_bundle: Rays to generate samples for
            render_step_size: Minimum step size to use for rendering
            near_plane: Near plane for raymarching
            far_plane: Far plane for raymarching
            alpha_thre: Opacity threshold skipping samples.
            cone_angle: Cone angle for raymarching, set to 0 for uniform marching.

        Returns:
            a tuple of (ray_samples, packed_info, ray_indices)
            The ray_samples are packed, only storing the valid samples.
            The ray_indices contains the indices of the rays that each sample belongs to.
        """

        rays_o = ray_bundle.origins.contiguous()
        rays_d = ray_bundle.directions.contiguous()
        times = ray_bundle.times

        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)

        else:
            t_min = None
            t_max = None

        if far_plane is None:
            far_plane = 1e10

        if ray_bundle.camera_indices is not None:
            camera_indices = ray_bundle.camera_indices.contiguous()
        else:
            camera_indices = None
        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=t_min,
            t_max=t_max,
            sigma_fn=self.get_sigma_fn(rays_o, rays_d, ray_bundle.pixel_area.contiguous(), times),
            render_step_size=render_step_size,
            near_plane=near_plane,
            far_plane=far_plane,
            stratified=self.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        num_samples = starts.shape[0]
        if num_samples == 0:
            # create a single fake sample and update packed_info accordingly
            # this says the last ray in packed_info has 1 sample, which starts and ends at 1
            ray_indices = torch.zeros((1,), dtype=torch.long, device=rays_o.device)
            starts = torch.ones((1,), dtype=starts.dtype, device=rays_o.device)
            ends = torch.ones((1,), dtype=ends.dtype, device=rays_o.device)

        origins = rays_o[ray_indices]
        dirs = rays_d[ray_indices]
        if camera_indices is not None:
            camera_indices = camera_indices[ray_indices]

        ray_samples = RaySamples(
            frustums=Frustums(
                origins=origins,
                directions=dirs,
                starts=starts[..., None],
                ends=ends[..., None],
                pixel_area=ray_bundle[ray_indices].pixel_area,
            ),
            camera_indices=camera_indices,
        )

        if ray_bundle.times is not None:
            ray_samples.times = ray_bundle.times[ray_indices]

        if ray_bundle.metadata is not None:
            ray_samples.metadata = {}
            for k, v in ray_bundle.metadata.items():
                if isinstance(v, torch.Tensor):
                    ray_samples.metadata[k] = v[ray_indices]

        return ray_samples, ray_indices
