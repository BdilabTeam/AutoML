from typing import Optional

from ..settings import Settings
from ..databases.mysql import MySQLServer
from ..errors import MySQLServerNotExistError

class DataPlane(object):
    """
    Internal implementation of handlers, used by REST servers.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        
        if self._settings.mysql_enabled:
            self._mysql_server = MySQLServer(self._settings).start()

    
    def get_sql_session(self):
        """Provide database session
        """
        if not self._mysql_server:
            raise MySQLServerNotExistError("No available MySQL database server")
        
        session_generator = self._mysql_server.get_session_generator()
        try:
            session = next(session_generator)
            return session
        finally:
            session.close()

