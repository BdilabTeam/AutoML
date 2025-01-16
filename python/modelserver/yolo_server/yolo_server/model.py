import os.path

# import tensorflow as tf
import numpy as np

from typing import Dict, Union
import logging

from tserve import Model
# from tserve.python.tserve.tserve import Model
from tserve import InferRequest, InferResponse
# from tserve.python.tserve.tserve import InferRequest, InferResponse
from tserve import get_predict_input, get_predict_response
# from tserve.python.tserve.tserve import get_predict_input, get_predict_response
from tserve import InferenceError
# from tserve.python.tserve.tserve import InferenceError

from ultralytics import YOLO

class YoloModel(Model):
    def __init__(self, name: str, pretrained_model_name_or_path: str):
        """
        An autokeras trained deployment model.
            Parameters:
            name (`str`):
                The name of a model.
            pretrained_model_name_or_path (`str`):
                The storage path of a model.
        """
        super().__init__(name)
        self.name = name
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model = None
        self.ready = False

    def load(self) -> bool:
        """
        Load an image-classification model.
            Parameters:
            pretrained_model_name_or_path (`str`):
                The storage path of a model.
        """

        # Load a model
        self.model = YOLO(model=self.pretrained_model_name_or_path)

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
            # inputs = get_predict_input(payload=payload)


            # payload = np.random.rand(640, 1280, 3).astype(np.uint8)


            preds = self.model.predict(source=payload)


            result = []

            for r in preds:
                print("probs:", r.probs)
                result.append(r.probs.data)

            result = result[0].numpy()

            return get_predict_response(payload=payload, result=result, model_name=self.name)
        except Exception as e:
            raise InferenceError(str(e))
