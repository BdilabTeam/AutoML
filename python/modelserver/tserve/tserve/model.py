from typing import Any, Union, List, Dict
from enum import Enum
import inspect
import time
import logging
import httpx
from httpx import HTTPStatusError
from .protocol.infer_type import InferRequest, InferResponse
from .errors import InvalidInput
import orjson


logger = logging.getLogger(__name__)


PREDICTOR_URL_FORMAT = "{0}://{1}/v1/models/{2}:predict"
PREDICTOR_V2_URL_FORMAT = "{0}://{1}/v2/models/{2}/infer"


def get_latency_ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 9)


class PredictorProtocol(Enum):
    REST_V1 = "v1"
    REST_V2 = "v2"
    GRPC_V2 = "grpc-v2"
    

def is_v2(protocol: PredictorProtocol) -> bool:
    return protocol != PredictorProtocol.REST_V1


class Model:
    def __init__(self, name: str):
        """
        Model is intended to be subclassed by various components within Treasures.
        
        Parameters:
            name (`str`): The name of the model.
        """
        self.name = name
        self.ready = False
        self.protocol = PredictorProtocol.REST_V1.value
        self.predictor_host = None
        self.timeout = 600
        self._http_client_instance = None
        self.enable_latency_logging = False
        self.use_ssl = False
        
    async def __call__(self, 
                       body: Union[Dict, Any], 
                       headers: Dict[str, str] = None) -> Dict:
        """
        Method to call model with the given input.

        Parameters:
            body (Dict|CloudEvent|InferRequest): Request payload body.
            headers (Dict): Request headers.

        Returns:
            Dict: Response output from preprocess -> predictor -> postprocess
        """
        request_id = headers.get("x-request-id", "N.A.") if headers else "N.A."
        
        # latency vars
        preprocess_ms = 0
        explain_ms = 0
        predict_ms = 0
        postprocess_ms = 0
        
        # 预处理
        start = time.time()
        payload = await self.preprocess(body, headers) if inspect.iscoroutinefunction(self.preprocess) else self.preprocess(body, headers)
        preprocess_ms = get_latency_ms(start, time.time())
        payload = self.validate(payload)
        
        # 预测
        start = time.time()
        response = (await self.predict(payload, headers)) if inspect.iscoroutinefunction(self.predict) else self.predict(payload, headers)
        predict_ms = get_latency_ms(start, time.time())
        
        # 后处理
        start = time.time()
        response = self.postprocess(response, headers)
        postprocess_ms = get_latency_ms(start, time.time())
        
        if self.enable_latency_logging is True:
            logger.info(f"requestId: {request_id}, preprocess_ms: {preprocess_ms}, "
                              f"predict_ms: {predict_ms}, "
                              f"postprocess_ms: {postprocess_ms}")

        return response
    
    @property
    def _http_client(self):
        if self._http_client_instance is None:
            self._http_client_instance = httpx.AsyncClient()
        return self._http_client_instance
    
    def validate(self, payload):
        # TODO: validate the request if self.get_input_types() defines the input types.
        if self.protocol == PredictorProtocol.REST_V2.value:
            if "inputs" in payload and not isinstance(payload["inputs"], list):
                raise InvalidInput("Expected \"inputs\" to be a list")
        elif self.protocol == PredictorProtocol.REST_V1.value:
            if isinstance(payload, Dict) and "instances" in payload and not isinstance(payload["instances"], list):
                raise InvalidInput("Expected \"instances\" to be a list")
        return payload
    
    def load(self) -> bool:
        """
        Load a model. Load handler can be overridden to load the model from storage.
        
        Returns:
            bool: True if model is ready, False otherwise
        """
        raise NotImplementedError
    
    async def preprocess(self, 
                         payload: Union[Dict, InferRequest], 
                         headers: Dict[str, str] = None) -> Union[Dict, InferRequest]:
        """
        `preprocess` handler can be overridden for data or feature transformation.
        The default implementation decodes to Dict if it is a binary CloudEvent
        or gets the data field from a structured CloudEvent.

        Parameters:
            payload (Dict|InferRequest): Body of the request, v2 endpoints pass InferRequest.
            headers (Dict): Request headers.

        Returns:
            Dict|InferRequest: Transformed inputs to ``predict`` handler or return InferRequest for predictor call.
        """

        return payload

    def postprocess(self, 
                    response: Union[Dict, InferResponse], 
                    headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        """
        The postprocess handler can be overridden for inference response transformation.

        Parameters:
            response (Dict|InferResponse|ModelInferResponse): The response passed from ``predict`` handler.
            headers (Dict): Request headers.

        Returns:
            Dict: post-processed response.
        """
        return response  
    
    async def _http_predict(self, 
                            payload: Union[Dict, InferRequest], 
                            headers: Dict[str, str] = None) -> Dict:
        protocol = "https" if self.use_ssl else "http"
        predict_url = PREDICTOR_URL_FORMAT.format(protocol, self.predictor_host, self.name)
        if self.protocol == PredictorProtocol.REST_V2.value:
            predict_url = PREDICTOR_V2_URL_FORMAT.format(protocol, self.predictor_host, self.name)

        # Adjusting headers. Inject content type if not exist.
        # Also, removing host, as the header is the one passed to transformer and contains transformer's host
        predict_headers = {'Content-Type': 'application/json'}
        if headers is not None:
            if 'x-request-id' in headers:
                predict_headers['x-request-id'] = headers['x-request-id']
            if 'x-b3-traceid' in headers:
                predict_headers['x-b3-traceid'] = headers['x-b3-traceid']
        if isinstance(payload, InferRequest):
            payload = payload.to_rest()
        data = orjson.dumps(payload)
        response = await self._http_client.post(
            predict_url,
            timeout=self.timeout,
            headers=predict_headers,
            content=data
        )
        if not response.is_success:
            message = (
                "{error_message}, '{0.status_code} {0.reason_phrase}' for url '{0.url}'"
            )
            error_message = ""
            if "content-type" in response.headers and response.headers["content-type"] == "application/json":
                error_message = response.json()
                if "error" in error_message:
                    error_message = error_message["error"]
            message = message.format(response, error_message=error_message)
            raise HTTPStatusError(message, request=response.request, response=response)
        return orjson.loads(response.content)

    async def predict(self, 
                      payload: Union[Dict, InferRequest],
                      headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        """
        Parameters:
            payload (Dict|InferRequest|ModelInferRequest): Prediction inputs passed from ``preprocess`` handler.
            headers (Dict): Request headers.

        Returns:
            Dict|InferResponse|ModelInferResponse: Return InferResponse for serializing the prediction result or
            return the response from the predictor call.
        """
        if not self.predictor_host:
            raise NotImplementedError("Could not find predictor_host.")
        
        #TODO support grpc
        res = await self._http_predict(payload, headers)
        # return an InferResponse if this is REST V2, otherwise just return the dictionary
        return InferResponse.from_rest(self.name, res) if is_v2(PredictorProtocol(self.protocol)) else res
