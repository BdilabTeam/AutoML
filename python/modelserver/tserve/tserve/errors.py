from http import HTTPStatus
from fastapi.responses import JSONResponse
from fastapi import status
import logging

logger = logging.getLogger(__name__)


class TServerError(Exception):
    def __init__(self, msg: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(msg)
        self.status_code = status_code

class ModelMissingError(Exception):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return self.path


class InferenceError(RuntimeError):
    def __init__(self, reason, status=None, debug_details=None):
        self.reason = reason
        self.status = status
        self.debug_details = debug_details

    def __str__(self):
        msg = super().__str__() if self.reason is None else self.reason
        if self.status is not None:
            msg = '[' + self.status + '] ' + msg
        return msg


class InvalidInput(ValueError):
    """
    Exception class indicating invalid input arguments.
    HTTP Servers should return HTTP_400 (Bad Request).
    """

    def __init__(self, reason):
        self.reason = reason

    def __str__(self):
        return self.reason


class ModelNotFound(Exception):
    """
    Exception class indicating requested model does not exist.
    HTTP Servers should return HTTP_404 (Not Found).
    """

    def __init__(self, model_name=None):
        self.reason = f"Model with name {model_name} does not exist."

    def __str__(self):
        return self.reason


class ModelNotReady(RuntimeError):
    def __init__(self, model_name: str, detail: str = None):
        self.model_name = model_name
        self.error_msg = f"Model with name {self.model_name} is not ready."
        if detail:
            self.error_msg = self.error_msg + " " + detail

    def __str__(self):
        return self.error_msg


async def exception_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content={"error": str(exc)})


async def invalid_input_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.BAD_REQUEST, content={"error": str(exc)})


async def inference_error_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content={"error": str(exc)})


async def generic_exception_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        content={"error": f"{type(exc).__name__} : {str(exc)}"})


async def model_not_found_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.NOT_FOUND, content={"error": str(exc)})


async def model_not_ready_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.SERVICE_UNAVAILABLE, content={"error": str(exc)})


async def not_implemented_error_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.NOT_IMPLEMENTED, content={"error": str(exc)})
