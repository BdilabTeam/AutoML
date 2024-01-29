class BaseTrainerConfig(object):
    model_type: str = ""
    def __init__(self, **kwargs) -> None:
        self.task_type = kwargs.pop("task_type", None)
        self.trainer_class_name = kwargs.pop("trainer_class_name", None)
    
    def to_dict(self):
        raise NotImplementedError