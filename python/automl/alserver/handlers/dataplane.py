from typing import Optional, Dict, Any, Callable
from collections import OrderedDict

from ..settings import Settings
from ..databases.mysql import MySQLServer
from ..errors import MySQLServerNotExistError
from ..operators import TrainingClient

NAMESPACE = 'zauto'



class DataPlane(object):
    """
    Internal implementation of handlers, used by REST servers.
    """

    def __init__(self, settings: Settings):
        
        if settings.mysql_enabled:
            self._mysql_server = MySQLServer(settings).start()

        if settings.kubernetes_enabled:
            self.training_client = TrainingClient(
                config_file=settings.kube_config_file, 
                context=settings.context,
                # client_configuration=settings.client_configuration    
            )
    
    def get_sql_session(self):
        """Provide database session
        """
        if not hasattr(self, '_mysql_server'):
            raise MySQLServerNotExistError("No available MySQL database server")
        
        session_generator = self._mysql_server.get_session_generator()
        try:
            session = next(session_generator)
            return session
        finally:
            session.close()
    
    def get_train_func(self, model_type: str):
        pass
    
    def get_host_ip(self, threshold):
        pass
    
    def get_default_namespace(self):
        return NAMESPACE
    
    def get_tfjob_params(self):
        pass
    
    def create_tfjob(
        self, 
        name: str, 
        func: Callable, 
        parameters: Optional[Dict[str, Any]],
        host_ip: Optional[str]
    ):
        self.training_client.create_tfjob_from_func(
            name=name,
            func=func,
            parameters=parameters,
            namespace=NAMESPACE,
            host_ip=host_ip
        )