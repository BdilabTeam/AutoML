from automl.databases.mysql import MySQLServer
from automl.databases.mysql.models import TrainingProject
from automl.settings import Settings
from automl.utils.logging import get_logger

from sqlalchemy.orm.session import Session
import pytest

logger = get_logger(__name__)

class TestMysqlServer:
    @pytest.fixture
    def session(self):
        settings = Settings()
        mysql_server = MySQLServer(settings=settings)
        mysql_server.start()
        session_generator = mysql_server.get_session_generator()
        return next(session_generator)
   
    def test_get_session(self, session: Session):
        training_project = session.query(TrainingProject).filter(TrainingProject.id == 5).one()
        print(training_project)