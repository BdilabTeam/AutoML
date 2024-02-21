from .settings import Settings
from .server import AutoMLServer
from .utils.logging import get_logger
import asyncio

logger = get_logger()
async def main():
    settings = Settings(
        kubernetes_enabled=True,
        model_selection_enabled=True,
        mysql_enabled=True,
        http_port=31185
    )
    logger.info(f"The parameters of the AutoML-Server: \n{settings.__str__()}")
    
    server = AutoMLServer(settings)
    await server.start()

if __name__=="__main__":
    asyncio.run(main())