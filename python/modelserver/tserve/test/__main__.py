import argparse
import logging

import tserve
from tserve.errors import ModelMissingError

from .custom_model_repository import CustomModelRepository
from .custom_model import CustomModel

logger = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "model"
DEFAULT_LOCAL_MODEL_DIR = "/Users/treasures/Downloads"

parser = argparse.ArgumentParser(parents=[tserve.model_server.parser])
parser.add_argument('--model_dir', required=False, default=DEFAULT_LOCAL_MODEL_DIR,
                    help='A URI pointer to the model binary')
parser.add_argument('--model_name', required=False, default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = CustomModel(args.model_name, args.model_dir)
    try:
        model.load()
    except ModelMissingError:
        logging.error(f"fail to locate model file for model {args.model_name} under dir {args.model_dir},"
                      f"trying loading from model repository.")

    tserve.model_server.ModelServer(registered_models=CustomModelRepository(args.model_dir)).start(
        [model] if model.ready else [])