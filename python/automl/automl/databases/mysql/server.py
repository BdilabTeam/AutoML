from typing import Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    sessionmaker,
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

class MySQLServer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

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
    
    async def start(self):
        # TODO 哪一步可以await?
        pymysql.install_as_MySQLdb()
        database_config = self._get_config()
        database_url = URL(**database_config)
        logger.info(
            "MySQL server running on "
            f"http://{self._settings.host_mysql}:{self._settings.port_mysql}"
        )
        if self._settings.async_enabled:
            self._engine = create_async_engine(database_url, echo=True)
            self._SessionLocal = async_sessionmaker(engine=self._engine, expire_on_commit=False)
        else:
            self._engine = create_engine(database_url, echo=True)
            self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
    
    async def stop(self):
        # TODO 哪一步可以await?
        self._SessionLocal.close()
        self._engine.dispose()
    
    def get_session_generator(self):
        session = self._SessionLocal()
        try:
            yield session
        finally:
            session.close()