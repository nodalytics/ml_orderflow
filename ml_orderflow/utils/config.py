import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "params.yaml"):
        self.config_path = Path(config_path)
        self.params = self._load_params()

    def _load_params(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def base(self):
        return self.params.get("base", {})

    @property
    def mlflow(self):
        return self.params.get("mlflow", {})

    @property
    def preprocessing(self):
        return self.params.get("preprocessing", {})

    @property
    def train(self):
        return self.params.get("train", {})

    @property
    def settings_dir(self):
        return Path(__file__).parent.parent / "settings"

    @property
    def logs_dir(self):
        logs = Path("logs")
        logs.mkdir(exist_ok=True)
        return logs

    @property
    def LOG_CONFIG_FILENAME(self):
        return "logging_config.yaml"

settings = Config()
