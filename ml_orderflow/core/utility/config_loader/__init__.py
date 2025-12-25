from pydantic.dataclasses import dataclass

from skincare_ai.core.utility.config_loader.read_json import JsonConfigReader
from skincare_ai.core.utility.config_loader.read_yaml import YamlConfigReader


@dataclass
class ConfigReaderInstance:
    json = JsonConfigReader()
    yaml = YamlConfigReader()
