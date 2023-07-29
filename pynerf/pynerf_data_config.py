"""
PyNeRF dataparsers configuration file.
"""
from nerfstudio.plugins.registry_dataparser import DataParserSpecification

from pynerf.data.dataparsers.adop_dataparser import AdopDataParserConfig
from pynerf.data.dataparsers.mipnerf_dataparser import MipNerf360DataParserConfig
from pynerf.data.dataparsers.multicam_dataparser import MulticamDataParserConfig

multicam_dataparser = DataParserSpecification(config=MulticamDataParserConfig())
mipnerf360_dataparser = DataParserSpecification(config=MipNerf360DataParserConfig())
adop_dataparser = DataParserSpecification(config=AdopDataParserConfig())
