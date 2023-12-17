from typing import Optional
from ..settings import Settings


class DataPlane(object):
    """
    Internal implementation of handlers, used by REST servers.
    """

    def __init__(self, settings: Settings):
        self._settings = settings

