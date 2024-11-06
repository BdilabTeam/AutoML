from typing import Optional
from fastapi import Request
from pydantic import BaseModel

from .responses import Response
from ..errors import AutoMLServerError


class APIErrorResponse(BaseModel):
    error: Optional[str] = None


async def handle_server_error(request: Request, exc: AutoMLServerError) -> Response:
    err_res = APIErrorResponse(error=str(exc))
    return Response(status_code=exc.status_code, content=err_res.dict())


_EXCEPTION_HANDLERS = {AutoMLServerError: handle_server_error}