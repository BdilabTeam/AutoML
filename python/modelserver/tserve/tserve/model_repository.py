from typing import Dict, Optional, Union
from .model import Model
import os

MODEL_MOUNT_DIRS = "/mnt/models"


class ModelRepository:
    """
    Model repository interface.

    It follows NVIDIA Triton's `Model Repository Extension`_.

    .. _Model Repository Extension:
        https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md
    """

    def __init__(self, models_dir: str = MODEL_MOUNT_DIRS):
        self.models: Dict[str, Model] = {}
        self.models_dir = models_dir

    def load_models(self):
        for name in os.listdir(self.models_dir):
            d = os.path.join(self.models_dir, name)
            if os.path.isdir(d):
                self.load_model(name)

    def set_models_dir(self, models_dir):  # used for unit tests
        self.models_dir = models_dir

    def get_model(self, name: str) -> Optional[Model]:
        return self.models.get(name, None)

    def get_models(self) -> Dict[str, Model]:
        return self.models

    def is_model_ready(self, name: str):
        model = self.get_model(name)
        if not model:
            return False
        if isinstance(model, Model):
            return False if model is None else model.ready
        else:
            return True

    def update(self, model: Model):
        self.models[model.name] = model

    def load(self, name: str) -> bool:
        pass

    def load_model(self, name: str) -> bool:
        pass

    def unload(self, name: str):
        if name in self.models:
            del self.models[name]
        else:
            raise KeyError(f"model {name} does not exist")