import json
from abc import ABC
from pathlib import Path

from skincare_ai.utils.config import settings
from skincare_ai.core.utility.config_loader.config_interface import ConfigReaderInterface
from skincare_ai.core.utility.config_loader.serializer import Struct


class JsonConfigReader(ConfigReaderInterface, ABC):

    def __init__(self):
        super(JsonConfigReader, self).__init__()

    def read_config_from_file(self, config_filename: str):
        conf_path = settings.settings_dir / config_filename
        with open(conf_path) as file:
            config = json.load(file)
        config_object = Struct(**config)
        return config_object
