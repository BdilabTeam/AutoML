from sqlalchemy import create_engine
from sqlalchemy.orm import (
    sessionmaker,
    Session
)
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine
)
from sqlalchemy.engine.url import URL

import pymysql

from ...settings import Settings
from ...utils.logging import get_logger

logger = get_logger(__name__)

class MySQLClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._initialize()

    def _get_config(self) -> dict:
        DATABASE_CONFIG = {}
        DATABASE_CONFIG.update(
            {
                "drivername": self._settings.drivername_mysql,
                "host": self._settings.host_mysql,
                "port": self._settings.port_mysql,
                "username": self._settings.username_mysql,
                "password": self._settings.password_mysql,
                "database": self._settings.database_mysql,
                "query": self._settings.query_mysql
            }
        )
        return DATABASE_CONFIG
    
    def _initialize(self):
        pymysql.install_as_MySQLdb()
        database_config = self._get_config()
        database_url = URL(**database_config)
        logger.info(
            "MySQLdb running on "
            f"http://{self._settings.host_mysql}:{self._settings.port_mysql}"
        )
        if self._settings.async_enabled:
            self._engine = create_async_engine(database_url, echo=True)
            self._SessionLocal = async_sessionmaker(engine=self._engine, expire_on_commit=False)
        else:
            self._engine = create_engine(database_url, echo=True)
            self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
    
    def stop(self):
        self._SessionLocal.close()
        self._engine.dispose()
    
    def get_session_generator(self):
        session = self._SessionLocal()
        try:
            yield session
        finally:
            logger.info("Close session object")
            session.close()
    
    def get_session(self) -> Session:
        return next(self.get_session_generator())
    
    def generate_schemas(self) -> bool:
        from ...models.base import Base
        Base.metadata.create_all(self._engine)
        return True