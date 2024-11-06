import logging
import os


from tserve import ModelRepository, MODEL_MOUNT_DIRS
from yolo_server.model import YoloModel



class YoloModelRepository(ModelRepository):

    def __init__(self, model_dir: str = MODEL_MOUNT_DIRS):
        super().__init__(model_dir)
        standardized_path = model_dir.replace('\\', os.sep)
        path_parts = standardized_path.split(os.sep)
        if len(path_parts) > 1:
            self.load_model(path_parts[-2])
        else:
            logging.error("Can't load model, because the directory is too short")

    async def load(self, name: str) -> bool:
        return self.load_model(name)

    def load_model(self, name: str) -> bool:
        model = YoloModel(name=name, pretrained_model_name_or_path=self.models_dir)
        # model = DataClassifierModel(name=name, pretrained_model_name_or_path=os.path.join(self.models_dir, name))
        if model.load():
            self.update(model)
        return model.ready
