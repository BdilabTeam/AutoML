import pytest

from autotrain.trainings.utils import AutoArgumentParser
from autotrain.trainings.utils import TrainingArguments

class TestAutoArgumentParser:
    @pytest.fixture
    def auto_argument_parser(self):
        return AutoArgumentParser(TrainingArguments)
    
    def test_parser_dict(self, parser: AutoArgumentParser):
        args = {
            "task_type": "structured_data_regression",
            "model_type": "densenet"
        }
        train_args, = parser.parse_dict(args=args)
        print(train_args)
        print(f"task_type: {train_args.task_type}")