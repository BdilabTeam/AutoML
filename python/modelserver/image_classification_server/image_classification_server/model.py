from tserve import Model
from transformers import AutoModelForImageClassification, pipeline, AutoImageProcessor
from typing import Dict, Union
import logging
from tserve import InferRequest, InferResponse
from tserve import get_predict_input, get_predict_response
from tserve import InferenceError

logger = logging.getLogger(__name__)


class ImageClassificationModel(Model):
    
    def __init__(self, name: str, pretrained_model_name_or_path: str) -> None:
        """
        An image-classification deployment model.
        
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
        
        Parameters:
            pretrained_model_name_or_path (`str`):
                The storage path of a model.
        """
        # im_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path)
        # model = AutoModelForImageClassification.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name_or_path)
        self.pipeline = pipeline(
            task="image-classification", 
            # model=model,
            model=self.pretrained_model_name_or_path,
            # image_processor=im_processor,
            # image_processor=self.pretrained_model_name_or_path,
            tokenizer=self.pretrained_model_name_or_path,
            device_map="auto",
            framework="pt"
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