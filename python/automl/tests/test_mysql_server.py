from alserver.databases.mysql import MySQLClient
from alserver.models.experiment import Experiment
from alserver.settings import Settings

from sqlalchemy.orm.session import Session
import pytest


class TestMySQLClient:
    @pytest.fixture
    def mysql_client(self):
        settings = Settings()
        return MySQLClient(settings=settings)
        
    @pytest.fixture
    def session(self, mysql_client: MySQLClient):
        session_generator = mysql_client.get_session_generator()
        return next(session_generator)
   
    def test_generate_schemas(self, mysql_client: MySQLClient):
        assert mysql_client.generate_schemas() == True, 'Schemas生成失败'
        
    def test_get_session(self, session: Session):
        experiment = session.query(Experiment).filter(Experiment.id == 1).one()
        print(experiment)