from typing import Optional, Union, Dict, List

from fastapi import Request, Response
from ...model_repository import ModelRepository
from ..infer_type import InferInput, InferOutput, InferRequest, InferResponse
from ...errors import ModelNotReady
from ..dataplane import DataPlane


class Endpoints(object):
    def __init__(self, 
                 model_repository: ModelRepository,
                 dataplane: DataPlane):
        self.model_repository = model_repository
        self.dataplane = dataplane
    
    async def infer(self,
                    raw_request: Request,
                    raw_response: Response,
                    model_name: str,
                    request_body: Dict,
                    model_version: Optional[str] = None):
        """
        Infer handler.

        Parameters:
            raw_request (Request): fastapi request object,
            raw_response (Response): fastapi response object,
            model_name (str): Model name.
            request_body (InferenceRequest): Inference request body.
            model_version (Optional[str]): Model version (optional).

        Returns:
            InferenceResponse: Inference response object.
        """
        model_ready = self.dataplane.model_ready(model_name)

        if not model_ready:
            raise ModelNotReady(model_name)

        request_headers = dict(raw_request.headers)
        response, response_headers = await self.dataplane.infer(model_name=model_name,
                                                                request=request_body,
                                                                headers=request_headers)

        return response
        
    
    