"""
Weighted dataset.
"""

from typing import Dict

import numpy as np
import torch
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from pynerf.pynerf_constants import WEIGHT, TRAIN_INDEX


class WeightedDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self.weights = self.metadata.get(WEIGHT, None)
        self.train_indices = self.metadata.get(TRAIN_INDEX, None)
        self.depth_images = self.metadata.get("depth_image", None)

    def get_metadata(self, data: Dict) -> Dict:
        metadata = {}
        if self.weights is not None:
            metadata[WEIGHT] = torch.full(data["image"].shape[:2], float(self.weights[data["image_idx"]]))
        if self.train_indices is not None:
            metadata[TRAIN_INDEX] = torch.full(data["image"].shape[:2], int(self.train_indices[data["image_idx"]]),
                                                dtype=torch.long)

        if self.depth_images is not None:
            filepath = self.depth_images[data["image_idx"]]
            depth_image = torch.FloatTensor(np.load(filepath)).unsqueeze(-1) / self.metadata["pose_scale_factor"]
            metadata["depth_image"] = depth_image

        return metadata
