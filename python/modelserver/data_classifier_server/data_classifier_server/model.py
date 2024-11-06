import tensorflow as tf
import numpy as np
import autokeras as ak
from typing import Dict, Union
import logging

from tserve import Model
from tserve import InferRequest, InferResponse
from tserve import get_predict_input, get_predict_response
from tserve import InferenceError
class DataClassifierModel(Model):
    def __init__(self, name: str, pretrained_model_name_or_path: str):
        """
        An Structured-data-classification deployment model.
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
        self.model = tf.keras.models.load_model(self.pretrained_model_name_or_path, custom_objects=ak.CUSTOM_OBJECTS)

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
            inputs = get_predict_input(payload = payload)
            # print(inputs)
            # inputs = [[str(item) for item in inner_list] for inner_list in inputs]
            # print(inputs)
            result  = self.model.predict(inputs)
            return get_predict_response(payload=payload, result=result, model_name=self.name)
        except Exception as e:
            raise InferenceError(str(e))

