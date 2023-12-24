import sys
sys.path.append("/Users/treasures/Desktop/AutoML/python/automl")

import asyncio

from automl.databases.mysql import MySQLServer
from automl.databases.mysql.models import TrainingProject
from automl.settings import Settings
from automl.utils.logging import get_logger

logger = get_logger(__name__)

async def main():
    settings = Settings()
    mysql_server = MySQLServer(settings=settings)
    await mysql_server.start()
    session_generator = mysql_server.get_session_generator()
    try:
        session = next(session_generator)
    finally:
        session.close()
    training_project = session.query(TrainingProject).filter(TrainingProject.id == 5).one()
    print(training_project)

if __name__ == "__main__":
    asyncio.run(main())