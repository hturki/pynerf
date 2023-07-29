from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader

from pynerf.pynerf_constants import TRAIN_INDEX, WEIGHT


class WeightedFixedIndicesEvalDataloader(FixedIndicesEvalDataloader):

    def __next__(self):
        ray_bundle, batch = super().__next__()
        metadata = self.input_dataset._dataparser_outputs.metadata

        camera_indices = ray_bundle.camera_indices
        if TRAIN_INDEX in metadata:
            if ray_bundle.metadata is None:
                ray_bundle.metadata = {}

            ray_bundle.metadata[TRAIN_INDEX] = metadata[TRAIN_INDEX].to(camera_indices.device)[camera_indices]

        if WEIGHT in metadata:
            batch[WEIGHT] = metadata[WEIGHT].to(camera_indices.device)[camera_indices]

        return ray_bundle, batch
