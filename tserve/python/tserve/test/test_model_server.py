import sys
from pathlib import Path
import argparse
import logging
import tserve
import os
from typing import Dict, Union
from tserve import InferRequest, InferResponse
from tserve.utils.utils import get_predict_input, get_predict_response
from tserve.errors import InferenceError
from transformers import AutoModelForImageClassification, pipeline, AutoImageProcessor
from tserve.errors import ModelMissingError
from tserve.model_repository import ModelRepository, MODEL_MOUNT_DIRS
from tserve.model import Model




class ImageClassificationModelRepository(ModelRepository):

    def __init__(self, model_dir: str = MODEL_MOUNT_DIRS):
        super().__init__(model_dir)
        self.load_models()

    async def load(self, name: str) -> bool:
        return self.load_model(name)

    def load_model(self, name: str) -> bool:
        model = ImageClassificationModel(name=name, pretrained_model_name_or_path=os.path.join(self.models_dir, name))
        if model.load():
            self.update(model)
        return model.ready


class ImageClassificationModel(Model):

    def __init__(self, name: str, pretrained_model_name_or_path: str) -> None:
        """
        An image-classification model.

        Parameters:
            name (`str`):
                The name of a model.
            pretrained_model_name_or_path (`str`):
                The storage path of a model.
        """
        super().__init__(name)
        self.name = name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pipeline = None
        self.ready = False

    def load(self) -> bool:
        """
        Load an image-classification model.
        """
        im_processor = AutoImageProcessor.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path)
        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path)
        self.pipeline = pipeline(
            task="image-classification",
            model=model,
            image_processor=im_processor,
            device_map="auto"
        )
        self.ready = True
        return self.ready

    async def predict(self,
                      payload: Union[Dict, InferRequest],
                      headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        """
        Execute inference.

        Parameters:
            payload (`Union[Dict, InferRequest]`):
                The inputs of the model.
            headers (`Dict[str, str]`)
                The headers of the request.
        """
        try:
            inputs = get_predict_input(payload=payload)
            result = self.pipeline(inputs[0])
            return get_predict_response(payload=payload, result=result, model_name=self.name)
        except Exception as e:
            raise InferenceError(str(e))


DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/Users/treasures/Downloads/model"

parser = argparse.ArgumentParser(parents=[tserve.model_server.parser])
parser.add_argument('--model_dir', required=False, default=DEFAULT_LOCAL_MODEL_DIR,
                    help='A URI pointer to the model binary')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = ImageClassificationModel(args.model_name, args.model_dir)
    print(args.model_dir)
    try:
        model.load()
    except ModelMissingError:
        logging.error(f"fail to locate model file for model {args.model_name} under dir {args.model_dir},"
                      f"trying loading from model repository.")

    tserve.model_server.ModelServer(registered_models=ImageClassificationModelRepository(args.model_dir)).start(
        [model] if model.ready else [])