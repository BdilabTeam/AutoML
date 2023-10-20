from ..model_repository import ModelRepository
from ..errors import ModelNotFound
from typing import Union, Dict, Tuple, Optional
from ..model import Model
from .infer_type import InferRequest, InferResponse


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