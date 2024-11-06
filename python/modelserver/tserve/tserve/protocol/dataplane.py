from ..model_repository import ModelRepository
from ..errors import InvalidInput, ModelNotFound
from typing import Union, Dict, Tuple, Optional
from ..model import Model
from .infer_type import InferRequest, InferResponse
from ..utils.utils import create_response_cloudevent, is_structured_cloudevent
import cloudevents.exceptions as ce
from cloudevents.http import CloudEvent, from_http
from cloudevents.sdk.converters.util import has_binary_headers
import time
import orjson
import logging


JSON_HEADERS = ["application/json", "application/cloudevents+json", "application/ld+json"]

class DataPlane:
    """
    Model DataPlane
    """
    def __init__(self, model_registry: ModelRepository) -> None:
        self._model_registry = model_registry
    
    @property
    def model_registry(self):
        return self._model_registry

    def get_model_from_registry(self, name: str) -> Model:
        model = self._model_registry.get_model(name)
        if model is None:
            raise ModelNotFound(name)

        return model

    def get_model(self, name: str) -> Model:
        """
        Get the model instance with the given name.

        The instance can be either ``Model`` or ``RayServeHandle``.

        Parameters:
            name (str): Model name.

        Returns:
            Model|RayServeHandle: Instance of the model.
        """
        model = self._model_registry.get_model(name)
        if model is None:
            raise ModelNotFound(name)
        if not self._model_registry.is_model_ready(name):
            model.load()
        return model

    def model_ready(self, model_name: str) -> bool:
        """
        Check if a model is ready.

        Parameters:
            model_name (str): name of the model

        Returns:
            bool: True if the model is ready, False otherwise.

        Raises:
            ModelNotFound: exception if model is not found
        """
        if self._model_registry.get_model(model_name) is None:
            raise ModelNotFound(model_name)

        return self._model_registry.is_model_ready(model_name)
    
    @staticmethod
    def get_binary_cloudevent(body: Union[str, bytes, None], headers: Dict[str, str]) -> CloudEvent:
        """Helper function to parse CloudEvent body and headers.

        Args:
            body (str|bytes|None): Request body.
            headers (Dict[str, str]): Request headers.

        Returns:
            CloudEvent: A CloudEvent instance parsed from http body and headers.

        Raises:
            InvalidInput: An error when CloudEvent failed to parse.
        """
        try:
            # Use default unmarshaller if contenttype is set in header
            if "ce-contenttype" in headers:
                event = from_http(headers, body)
            else:
                event = from_http(headers, body, lambda x: x)

            return event
        except (ce.MissingRequiredFields, ce.InvalidRequiredFields, ce.InvalidStructuredJSON,
                ce.InvalidHeadersFormat, ce.DataMarshallerError, ce.DataUnmarshallerError) as e:
            raise InvalidInput(f"Cloud Event Exceptions: {e}")

    
    def decode(self, body, headers) -> Tuple[Union[Dict, InferRequest], Dict]:
        t1 = time.time()
        attributes = {}
        if isinstance(body, InferRequest):
            return body, attributes
        if headers:
            if has_binary_headers(headers):
                # returns CloudEvent
                body = self.get_binary_cloudevent(body, headers)
            elif "content-type" in headers and headers["content-type"] not in JSON_HEADERS:
                return body, attributes
            else:
                if type(body) is bytes:
                    try:
                        body = orjson.loads(body)
                    except orjson.JSONDecodeError as e:
                        raise InvalidInput(f"Unrecognized request format: {e}")
        elif type(body) is bytes:
            try:
                body = orjson.loads(body)
            except orjson.JSONDecodeError as e:
                raise InvalidInput(f"Unrecognized request format: {e}")

        decoded_body, attributes = self.decode_cloudevent(body)
        t2 = time.time()
        logging.debug(f"decoded request in {round((t2 - t1) * 1000, 9)}ms")
        return decoded_body, attributes

    def decode_cloudevent(self, body) -> Tuple[Union[Dict, InferRequest], Dict]:
        decoded_body = body
        attributes = {}
        if isinstance(body, CloudEvent):
            attributes = body._get_attributes()
            decoded_body = body.get_data()
            try:
                decoded_body = orjson.loads(decoded_body.decode('UTF-8'))
            except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
                # If decoding or parsing failed, check if it was supposed to be JSON UTF-8
                if "content-type" in body._attributes and \
                        (body._attributes["content-type"] == "application/cloudevents+json" or
                         body._attributes["content-type"] == "application/json"):
                    raise InvalidInput(f"Failed to decode or parse binary json cloudevent: {e}")

        elif isinstance(body, dict):
            if is_structured_cloudevent(body):
                decoded_body = body["data"]
                attributes = body
                del attributes["data"]
        return decoded_body, attributes

    def encode(self, model_name, response, headers, req_attributes: Dict) -> Tuple[Dict, Dict[str, str]]:
        response_headers = {}
        # if we received a cloudevent, then also return a cloudevent
        is_cloudevent = False
        is_binary_cloudevent = False
        if isinstance(response, InferResponse):
            response = response.to_rest()
        if headers:
            if has_binary_headers(headers):
                is_cloudevent = True
                is_binary_cloudevent = True
            if headers.get("content-type", "") == "application/cloudevents+json":
                is_cloudevent = True
        if is_cloudevent:
            response_headers, response = create_response_cloudevent(model_name, response, req_attributes,
                                                                    is_binary_cloudevent)

            if is_binary_cloudevent:
                response_headers["content-type"] = "application/json"
            else:
                response_headers["content-type"] = "application/cloudevents+json"
        return response, response_headers

    async def infer(self,
                    model_name: str,
                    request: Union[Dict, InferRequest],
                    headers: Optional[Dict[str, str]] = None) -> Tuple[Union[Dict, InferResponse], Dict[str, str]]:
        """
        Performs inference on the specified model with the provided body and headers.

        Parameters:
            model_name (str): Model name.
            request (bytes|Dict): Request body data.
            headers: (Optional[Dict[str, str]]): Request headers.

        Returns:
            Tuple[Union[str, bytes, Dict], Dict[str, str]]:
                - response: The inference result.
                - response_headers: Headers to construct the HTTP response.
        """
        # call model locally or remote model workers
        model = self.get_model(model_name)
        response = await model(request, headers=headers)

        return response, headers