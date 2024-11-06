import os
from tserve import ModelRepository, MODEL_MOUNT_DIRS
from .model import DataRegressionModel


class DataRegressionModelRepository(ModelRepository):

    def __init__(self, model_dir: str = MODEL_MOUNT_DIRS):
        super().__init__(model_dir)
        self.load_model("data_regression_best_model")

    async def load(self, name: str) -> bool:
        return self.load_model(name)

    def load_model(self, name: str) -> bool:
        model = DataRegressionModel(name=name, pretrained_model_name_or_path=self.models_dir)
        # model = DataRegressionModel(name=name, pretrained_model_name_or_path=os.path.join(self.models_dir, name))
        if model.load():
            self.update(model)
        return model.ready