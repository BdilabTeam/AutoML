import pytest
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional
from autotrain.utils import AutoArgumentParser

@dataclass
class TrainerArguments:
    model_type: str = field(default='resnet')
    task_type: str = field(default='')
    dp_image_size: Optional[Tuple[float]] = None
    mp_rotation_factor: Optional[List[float]] = None

class TestAutoArgumentParser:
    @pytest.fixture
    def trainer_arguments_parser(self):
        return AutoArgumentParser(TrainerArguments)
    
    def test_parser_dict_v1(self, trainer_arguments_parser: AutoArgumentParser):
        args_dict = {
            "task_type": "structured_data_regression",
            "model_type": "densenet"
        }
        trainer_args, = trainer_arguments_parser.parse_dict(args=args_dict)
        print(trainer_args)
    
    def test_parser_dict_v2(self, trainer_arguments_parser: AutoArgumentParser):
        @dataclass
        class TestType:
            dp_image_size: Tuple[float]
            mp_rotation_factor: List[float]

        t = TestType([1, 2], (3, 4))
        arg_dict = asdict(t)
        trainer_args, = trainer_arguments_parser.parse_dict(args=arg_dict)
        print(trainer_args)
    
    def test_parser_dict_v3(self, trainer_arguments_parser: AutoArgumentParser):
        arg_dict = {
            'dp_image_size': [1, 2],
            'mp_rotation_factor': (3, 4)
        }
        trainer_args, = trainer_arguments_parser.parse_dict(args=arg_dict)
        print(trainer_args)
        print(trainer_args.__dict__.items())
        print(getattr(trainer_args, 'task_type'))
        assert trainer_args.__dict__['dp_image_size'] == [1, 2]
        assert getattr(trainer_args, 'task_type') == ""