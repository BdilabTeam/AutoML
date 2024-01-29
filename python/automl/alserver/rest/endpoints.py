from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi import Body, File, UploadFile, Form
from typing_extensions import Annotated

from typing import Optional, List

from ..handlers import DataPlane
from ..schemas import input_schemas
import json


class Endpoints(object):
    """
    Implementation of REST endpoints.
    These take care of the REST/HTTP-specific things and then delegate the
    business logic to the internal handlers.
    """
    def __init__(self, data_plane: DataPlane):
        self._data_plane = data_plane
    
    # async def create_training_project(self, 
    #                          files: List[UploadFile] = File(description="Multiple files as UploadFile"), 
    #                          template_id: str = Form(),
    #                          template_parameters: str = Form()) -> JSONResponse:
    #        return JSONResponse(
    #         content="{'test': 'ok'}"
    #     )
    async def create_training_project(self, training_project = input_schemas.TrainingProjectCreate) -> JSONResponse:
        return JSONResponse(
        content="{'test': 'ok'}"
    )