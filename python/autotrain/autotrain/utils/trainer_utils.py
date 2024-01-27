from dataclasses import dataclass
from typing import Any

from .configuration_utils import BaseTrainerConfig

class BaseTrainerOutput(object):
    pass

class BaseTrainer(object):
    def __init__(self, config: BaseTrainerConfig) -> None:
        raise NotImplementedError
    def train(self, inputs: Any, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
    

