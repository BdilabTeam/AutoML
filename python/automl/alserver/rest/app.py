import os
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response as FastAPIResponse
from fastapi.routing import APIRoute as FastAPIRoute
from fastapi.middleware.cors import CORSMiddleware

from .endpoints import Endpoints
from .requests import Request
from .responses import Response
from .errors import _EXCEPTION_HANDLERS
from ..handlers import DataPlane

STATIC_FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata")

class APIRoute(FastAPIRoute):
    """
    Custom route to use our own Request handler.
    """

    def __init__(
        self,
        *args,
        response_model_exclude_unset=True,
        response_model_exclude_none=True,
        response_class=Response,
        **kwargs
    ):
        super().__init__(
            *args,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_none=response_model_exclude_none,
            response_class=response_class,
            **kwargs
        )

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> FastAPIResponse:
            request = Request(request.scope, request.receive)
            return await original_route_handler(request)

        return custom_route_handler

def create_app(
    data_plane: DataPlane,
) -> FastAPI:
    endpoints = Endpoints(data_plane)
    
    routes = [
        APIRoute(
            "/api/v1/selection/candidate-models",
            endpoints.get_candidate_models,
            methods=['POST'],
            tags=["model-selection"]
        ),
        APIRoute(
            "/api/v1/experiment",
            endpoints.create_experiment,
            methods=['POST'],
            tags=["experiment"]
        ),
        APIRoute(
            "/api/v1/experiment",
            endpoints.patch_experiment,
            methods=['PATCH'],
            tags=["experiment"]
        ),
        APIRoute(
            "/api/v1/experiment/overview/{experiment_name}",
            endpoints.get_experiment_overview,
            methods=['GET'],
            tags=["experiment"]
        ),
        APIRoute(
            "/api/v1/experiment/cards",
            endpoints.get_experiment_cards,
            methods=['GET'],
            tags=["experiment"]
        ),
        APIRoute(
            "/api/v1/experiment/{experiment_name}",
            endpoints.delete_experiment,
            methods=['DELETE'],
            tags=["experiment"]
        ),
        APIRoute(
            "/api/v1/experiment/evaluate",
            endpoints.evaluate_experiment,
            methods=['POST'],
            tags=["experiment"]
        ),
        APIRoute(
            "/api/v1/experiment/model/export/{experiment_name}",
            endpoints.export_best_model,
            methods=['GET'],
            tags=["experiment"]
        ),
        # APIRoute(
        #     "/api/v1/monitoring/info",
        #     endpoints.get_monitor_info,
        #     methods=['GET'],
        #     tags=["monitoring"]
        # ),
        APIRoute(
            "/api/v1/repository/model",
            endpoints.get_model_repository_info,
            methods=['GET'],
            tags=["model-repository"]
        ),
        APIRoute(
            "/api/v1/dataset/overview/{experiment_name}",
            endpoints.get_dataset_overview,
            methods=['GET'],
            tags=["dataset"]
        ),
        APIRoute(
            "/api/v1/platform/minio",
            endpoints.get_minio_url,
            methods=['GET'],
            tags=["platform"]
        ),
        APIRoute(
            "/api/v1/platform/data-annotation",
            endpoints.get_data_annotation_platform_url,
            methods=['GET'],
            tags=["platform"]
        ),
        APIRoute(
            "/api/v1/platform/task-types",
            endpoints.get_task_types,
            methods=['GET'],
            tags=["platform"]
        ),
        
    ]
    
    app = FastAPI(
        routes=routes,  # type: ignore
        default_response_class=Response,
        exception_handlers=_EXCEPTION_HANDLERS,  # type: ignore
    )
    app.router.route_class = APIRoute
    
    app.mount(
        path="/api/v1/metadata",
        app=StaticFiles(
            directory=STATIC_FILES_DIR
        ),
        name="metadata"
    )
    
    app.add_websocket_route(
        path="/api/v1/experiment/job/logs",
        route=endpoints.get_experiment_job_logs
    )
    

    return app