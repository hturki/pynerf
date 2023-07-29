from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Optional, Type, Dict

import torch
from PIL import Image
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
)
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipeline, DynamicBatchPipelineConfig
from nerfstudio.utils import profiler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


def get_average_eval_image_metrics(datamanager: DataManager, model: Model, step: Optional[int] = None,
                                   output_path: Optional[Path] = None, get_std: bool = False) -> Dict:
    """Same as in VanillaPipeline but removes the isinstance(self.datamanager, VanillaDataManager) assertion to handle
    RandomSubsetDataManager and can also handle the case where not every metrics_dict has the same keys (which is the
    case for metrics such as psnr_1.0)"""

    metrics_dict_list = []
    num_images = len(datamanager.fixed_indices_eval_dataloader)
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
        for camera_ray_bundle, batch in datamanager.fixed_indices_eval_dataloader:
            # time this the following line
            inner_start = time()
            height, width = camera_ray_bundle.shape
            num_rays = height * width
            outputs = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            metrics_dict, images_dict = model.get_image_metrics_and_images(outputs, batch)

            if output_path is not None:
                camera_indices = camera_ray_bundle.camera_indices
                assert camera_indices is not None
                for key, val in images_dict.items():
                    Image.fromarray((val * 255).byte().cpu().numpy()).save(
                        output_path / "{0:06d}-{1}.jpg".format(int(camera_indices[0, 0, 0]), key)
                    )
            assert "num_rays_per_sec" not in metrics_dict
            metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
            fps_str = "fps"
            assert fps_str not in metrics_dict
            metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
            metrics_dict_list.append(metrics_dict)
            progress.advance(task)
    # average the metrics list
    metrics_dict = {}
    metric_keys = set()
    for metrics_dict in metrics_dict_list:
        metric_keys.update(metrics_dict.keys())
    for key in metric_keys:
        if get_std:
            key_std, key_mean = torch.std_mean(
                torch.tensor([metrics_dict[key] for metrics_dict in filter(lambda x: key in x, metrics_dict_list)])
            )
            metrics_dict[key] = float(key_mean)
            metrics_dict[f"{key}_std"] = float(key_std)
        else:
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in
                                         filter(lambda x: key in x, metrics_dict_list)]))
            )

    return metrics_dict


@dataclass
class PyNeRFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: PyNeRFPipeline)
    """target class to instantiate"""


class PyNeRFPipeline(VanillaPipeline):

    @profiler.time_function
    def get_average_eval_image_metrics(
            self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        self.eval()
        metrics_dict = get_average_eval_image_metrics(self.datamanager, self.model, step, output_path, get_std)
        self.train()

        return metrics_dict


@dataclass
class PyNeRFDynamicBatchPipelineConfig(DynamicBatchPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: PyNeRFDynamicBatchPipeline)


class PyNeRFDynamicBatchPipeline(DynamicBatchPipeline):

    @profiler.time_function
    def get_average_eval_image_metrics(
            self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        self.eval()
        metrics_dict = get_average_eval_image_metrics(self.datamanager, self.model, step, output_path, get_std)
        self.train()

        return metrics_dict
