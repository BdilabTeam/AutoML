from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    sessionmaker,
    MappedAsDataclass,
    DeclarativeBase
)
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncAttrs,
    create_async_engine
)
from sqlalchemy.engine.url import URL

import pymysql

pymysql.install_as_MySQLdb()

# MySQL数据库 配置
DATABASE_CONFIG = {
    "drivername": "mysql+mysqldb",  # 或 "mysql+pymysql" 等，根据你的驱动程序
    "host": "124.70.188.119",  # 数据库服务器地址
    "port": 3307,  # 数据库端口号
    "username": "root",  # 数据库用户名
    "password": "bdilab@1308",  # 数据库密码
    "database": "automl",  # 要使用的数据库名
    "query": {"charset": "utf8mb4"}   # 可选的查询参数，如 {"charset": "utf8mb4"}
}
DATABASE_URL = URL(**DATABASE_CONFIG)

# 创建 MySQL 引擎
engine = create_engine(
    DATABASE_URL, echo=True
)


# 创建一个SessionLocal类, 用于开启数据库会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# # # 创建异步 MySQL 引擎
# async_engine = create_async_engine(
#     DATABASE_URL, echo=True
# )

# AsyncSessionLocal = async_sessionmaker(engine=engine, expire_on_commit=False)

# 创建一个Base类, 用于派生 SQLAlchemy 模型
# Base = declarative_base() 
class Base(MappedAsDataclass, DeclarativeBase):
    pass
