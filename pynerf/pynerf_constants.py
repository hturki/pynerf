from enum import Enum

EXPLICIT_LEVEL = 'explicit_level'
LEVELS = 'levels'
LEVEL_COUNTS = 'level_counts'

RGB = 'image'
DEPTH = 'depth'
RAY_INDEX = 'ray_index'
TRAIN_INDEX = 'train_index'
WEIGHT = 'weight'
POSE_SCALE_FACTOR = 'pose_scale_factor'
RENDER_LEVELS = 'render_levels'
class PyNeRFFieldHeadNames(Enum):
    LEVELS = 'levels'
    LEVEL_COUNTS = 'level_counts'
