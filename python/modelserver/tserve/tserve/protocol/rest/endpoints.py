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
                    request: Request,
                    model_name: str):
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
        
        body = await request.body()
        headers = dict(request.headers.items())
        infer_request, req_attributes = self.dataplane.decode(body=body,
                                                              headers=headers)
        response, response_headers = await self.dataplane.infer(model_name=model_name,
                                                                  request=infer_request,
                                                                  headers=headers)
        response, response_headers = self.dataplane.encode(model_name=model_name,
                                                           response=response,
                                                           headers=headers, req_attributes=req_attributes)

        if not isinstance(response, dict):
            return Response(content=response, headers=response_headers)
        return response
        
    
    