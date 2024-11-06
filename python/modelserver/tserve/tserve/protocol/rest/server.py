from fastapi import FastAPI, Request, Response
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute
from .endpoints import Endpoints
from ...errors import generic_exception_handler
from typing import List, Optional, Union, Dict
import socket
import uvicorn
import asyncio
from ...model_repository import ModelRepository
from ..dataplane import DataPlane


class _NoSignalUvicornServer(uvicorn.Server):
    def install_signal_handlers(self) -> None:
        pass


class RESTServer:
    def __init__(
        self, 
        model_repository: ModelRepository,
        dataplane: DataPlane
    ):
        self.model_repository = model_repository
        self.dataplane = dataplane
    
    def create_application(self) -> FastAPI:
        """
        Create ModelServer application with API routes.

        Returns:
            FastAPI: An application instance.
        """
        endpoints = Endpoints(model_repository=self.model_repository, dataplane=self.dataplane)
        return FastAPI(
            title="Treasures ModelServer",
            default_response_class=ORJSONResponse,
            routes=[
                APIRoute(r"/v2/models/{model_name}/infer", endpoint=endpoints.infer, methods=["POST"], response_model=None)
            ],
            exception_handlers={
                Exception: generic_exception_handler
            }
        )
    

class UvicornServer:
    def __init__(
        self,
        http_port: int,
        sockets: List[socket.socket],
        model_repository: ModelRepository,
        dataplane: DataPlane
    ):
        self.sockets = sockets
        rest_server = RESTServer(model_repository=model_repository, dataplane=dataplane)
        app = rest_server.create_application()
        self.cfg = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=http_port,
        )
        self.server = _NoSignalUvicornServer(self.cfg)
        
    def run_sync(self):
        server = uvicorn.Server(self.cfg)
        asyncio.run(server.serve(self.sockets))

    async def run(self):
        await self.server.serve()
        
    async def stop(self, sig: Optional[int] = None):
        if self.server:
            self.server.handle_exit(sig=sig, frame=None)