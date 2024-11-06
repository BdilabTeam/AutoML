import argparse
import logging

from .model import DataRegressionModel
from .dr_model_repository import DataRegressionModelRepository
from tserve.errors import ModelMissingError
import tserve

DEFAULT_MODEL_NAME = "data_regression_model"
DEFAULT_LOCAL_MODEL_DIR = "D:\\Project\\AM\\AutoML\\tserve\\autokeras\\autokeras\\structured_data_regressor\\best_model"

parser = argparse.ArgumentParser(parents=[tserve.model_server.parser])
parser.add_argument('--model_dir', required=False, default=DEFAULT_LOCAL_MODEL_DIR,
                    help='A URI pointer to the model binary')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
args, _ = parser.parse_known_args()

if __name__ == '__main__':
    model = DataRegressionModel(args.model_name, args.model_dir)
    try:
        model.load()
    except ModelMissingError:
        logging.error(f"fail to locate model file for model {args.model_name} under dir {args.model_dir},"
                      f"trying loading from model repository.")

    tserve.ModelServer(registered_models=DataRegressionModelRepository(args.model_dir)).start(
        [model] if model.ready else []
    )