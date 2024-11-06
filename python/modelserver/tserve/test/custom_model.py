from tserve.model import Model
from tserve import InferRequest, InferResponse
from tserve.utils.utils import get_predict_input, get_predict_response
from tserve.errors import InferenceError

import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)
class MockModel:
    def __init__(self) -> None:
        pass
    def __call__(self, inputs: str) -> Any:
        return inputs
        
class CustomModel(Model):

    def __init__(self, name: str, model_dir: str) -> None:
        """
        An image-classification model.

        Parameters:
            name (`str`):
                The name of a model.
        """
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.model = None
        self.ready = False

    def load(self) -> bool:
        """
        Load a model.
        """
        logger.info("加载模型中 ......")
        self.model = MockModel()
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
            logger.info(inputs)
            result = self.model(inputs[0])
            return get_predict_response(payload=payload, result=result, model_name=self.name)
        except Exception as e:
            raise InferenceError(str(e))