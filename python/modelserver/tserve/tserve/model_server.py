import argparse
from typing import Optional, List
from .model import Model
from .utils import utils
import asyncio
import concurrent.futures
import sys
import signal
import socket
from .protocol.rest.server import UvicornServer
import multiprocessing
from multiprocessing import Process
from .model_repository import ModelRepository
import logging
from .protocol.dataplane import DataPlane


logger = logging.getLogger(__name__)

DEFAULT_HTTP_PORT = 8080
DEFAULT_GRPC_PORT = 8081

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--http_port", default=DEFAULT_HTTP_PORT, type=int,
                    help="The HTTP Port listened to by the model tserve.")
parser.add_argument("--workers", default=1, type=int,
                    help="The number of workers for multi-processing.")
parser.add_argument("--max_threads", default=4, type=int,
                    help="The number of max processing threads in each worker.")
parser.add_argument('--max_asyncio_workers', default=None, type=int,
                    help='Max number of asyncio workers to spawn')
parser.add_argument("--enable_latency_logging", default=True, type=lambda x: utils.strtobool(x),
                    help="Output a log per request with latency metrics.")

args, _ = parser.parse_known_args()

DEFAULT_HTTP_PORT = 8080


class ModelServer(object):
    """
    ModelServer

    Parameters:
        http_port (int): HTTP port. Default: ``8080``.
        workers (int): Number of workers for uvicorn. Default: ``1``.
        max_threads (int): Max number of processing threads. Default: ``4``
        max_asyncio_workers (int): Max number of AsyncIO threads. Default: ``None``
        registered_models (ModelRepository): Model repository with registered models.
        enable_latency_logging (bool): Whether to log latency metric. Default: ``True``.
    """

    def __init__(self,
                 registered_models: ModelRepository = ModelRepository(),
                 http_port: int = args.http_port,
                 workers: int = args.workers,
                 max_threads: int = args.max_threads,
                 max_asyncio_workers: int = args.max_asyncio_workers,
                 enable_latency_logging: bool = args.enable_latency_logging):
        self.registered_models = registered_models
        self.http_port = http_port
        self.workers = workers
        self.max_threads = max_threads
        self.max_asyncio_workers = max_asyncio_workers
        self.enable_latency_logging = enable_latency_logging
        self.dataplane = DataPlane(model_registry=registered_models)
        
    def start(self, models: List[Model]):
        if isinstance(models, list):
            for model in models:
                if isinstance(model, Model):
                    self.register_model(model)
                    # pass whether to log request latency into the model
                    model.enable_latency_logging = self.enable_latency_logging
                else:
                    raise RuntimeError("Model type should be 'Model'")
        else:
            raise RuntimeError("Unknown model collection types")
        
        if self.max_asyncio_workers is None:
            self.max_asyncio_workers = min(32, utils.cpu_count() + 4)
        logger.info(f"Setting max asyncio worker threads as {self.max_asyncio_workers}")
        asyncio.get_event_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self.max_asyncio_workers)
        )
        
        async def serve():
            logger.info(f"Starting uvicorn with {self.workers} workers")
            loop = asyncio.get_event_loop()
            if sys.platform not in ['win32', 'win64']:
                sig_list = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT]
            else:
                sig_list = [signal.SIGINT, signal.SIGTERM]

            for sig in sig_list:
                #不加try_except代码块的话，windows执行报错，因为windows没有add_signal_handler方法的实现
                try:
                    loop.add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(self.stop(sig=s))
                )
                except NotImplementedError:
                    pass

            if self.workers == 1:
                self._rest_server = UvicornServer(
                    http_port=self.http_port, 
                    sockets=[],
                    model_repository=self.registered_models,
                    dataplane=self.dataplane
                )
                await self._rest_server.run()
            else:
                serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                serversocket.bind(('0.0.0.0', self.http_port))
                serversocket.listen(5)
                multiprocessing.set_start_method('fork')
                server = UvicornServer(
                    http_port=self.http_port, 
                    sockets=[serversocket],
                    model_repository=self.registered_models,
                    dataplane=self.dataplane
                )
                for _ in range(self.workers):
                    p = Process(target=server.run_sync)
                    p.start()

        async def servers_task():
            servers = [serve()]
            await asyncio.gather(*servers)

        asyncio.run(servers_task())
    
    async def stop(self, sig: Optional[int] = None):
        logger.info("Stopping the model tserve")
        if self._rest_server:
            logger.info("Stopping the rest tserve")
            await self._rest_server.stop()

    def register_model(self, model: Model):
        if not model.name:
            raise Exception(
                "Failed to register model, model.name must be provided.")
        self.registered_models.update(model)
        logger.info("Registering model: %s", model.name)
    