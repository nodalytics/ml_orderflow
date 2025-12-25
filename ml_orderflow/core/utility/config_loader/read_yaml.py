from pathlib import Path

import yaml

from skincare_ai.utils.config import settings
from skincare_ai.core.utility.config_loader.config_interface import ConfigReaderInterface
from skincare_ai.core.utility.config_loader.serializer import Struct
from skincare_ai.core.utility.logger.custom_logging import LogHandler


class YamlConfigReader(ConfigReaderInterface):

    def __init__(self):
        super(YamlConfigReader, self).__init__()

    def read_config_from_file(self, config_filename: str):
        conf_path = settings.settings_dir / config_filename
        with open(conf_path) as file:
            config = yaml.safe_load(file)
        config_object = Struct(**config)
        return config_object
