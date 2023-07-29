from typing import Dict

import torch
from nerfstudio.data.pixel_samplers import PixelSampler

from pynerf.pynerf_constants import WEIGHT, TRAIN_INDEX


class WeightedPixelSampler(PixelSampler):

    def collate_image_dataset_batch_list(self, batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
        device = batch["image"][0].device
        num_images = len(batch["image"])

        # only sample within the mask, if the mask is in the batch
        all_indices = []
        all_images = []
        all_weights = []
        all_train_indices = []

        if "mask" in batch:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape

                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch

                # TODO(hturki): Need to add unsqueeze(0) so that indices shape is 3 - is this a bug in Nerfstudio?
                indices = self.sample_method(
                    num_rays_in_batch, 1, image_height, image_width, mask=batch["mask"][i].unsqueeze(0), device=device
                )
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])

                if WEIGHT in batch:
                    all_weights.append(batch[WEIGHT][i][indices[:, 1], indices[:, 2]])
                if TRAIN_INDEX in batch:
                    all_train_indices.append(batch[TRAIN_INDEX][i][indices[:, 1], indices[:, 2]])
        else:
            num_rays_in_batch = num_rays_per_batch // num_images
            for i in range(num_images):
                image_height, image_width, _ = batch["image"][i].shape
                if i == num_images - 1:
                    num_rays_in_batch = num_rays_per_batch - (num_images - 1) * num_rays_in_batch
                indices = self.sample_method(num_rays_in_batch, 1, image_height, image_width, device=device)
                indices[:, 0] = i
                all_indices.append(indices)
                all_images.append(batch["image"][i][indices[:, 1], indices[:, 2]])
                if WEIGHT in batch:
                    all_weights.append(batch[WEIGHT][i][indices[:, 1], indices[:, 2]])
                if TRAIN_INDEX in batch:
                    all_train_indices.append(batch[TRAIN_INDEX][i][indices[:, 1], indices[:, 2]])

        indices = torch.cat(all_indices, dim=0)

        c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
        collated_batch = {
            key: value[c, y, x]
            for key, value in batch.items()
            if key not in {"image_idx", "image", "mask", WEIGHT, TRAIN_INDEX} and value is not None
        }

        collated_batch["image"] = torch.cat(all_images, dim=0)

        if WEIGHT in batch:
            collated_batch[WEIGHT] = torch.cat(all_weights, dim=0)
            assert collated_batch[WEIGHT].shape == (num_rays_per_batch,), collated_batch[WEIGHT].shape

        if TRAIN_INDEX in batch:
            collated_batch[TRAIN_INDEX] = torch.cat(all_train_indices, dim=0)
            assert collated_batch[TRAIN_INDEX].shape == (num_rays_per_batch,), collated_batch[TRAIN_INDEX].shape

        assert collated_batch["image"].shape[0] == num_rays_per_batch

        # Needed to correct the random indices to their actual camera idx locations.
        indices[:, 0] = batch["image_idx"][c]
        collated_batch["indices"] = indices  # with the abs camera indices

        if keep_full_image:
            collated_batch["full_image"] = batch["image"]

        return collated_batch
