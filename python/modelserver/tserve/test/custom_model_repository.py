from tserve.model_repository import ModelRepository, MODEL_MOUNT_DIRS
import logging
from .custom_model import CustomModel

logger = logging.getLogger(__name__)


class CustomModelRepository(ModelRepository):

    def __init__(self, model_dir: str = MODEL_MOUNT_DIRS):
        super().__init__(model_dir)
        self.load_models()

    async def load(self, name: str) -> bool:
        return self.load_model(name)

    def load_model(self, name: str) -> bool:
        model = CustomModel(name=name, model_dir=self.models_dir)
        logger.info("加载模型 ......")
        return model.ready