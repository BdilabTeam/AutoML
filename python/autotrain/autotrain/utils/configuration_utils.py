class BaseTrainerConfig(object):
    model_type: str = ""
    
    def __init__(self, **kwargs) -> None:
        self.task_type = kwargs.pop("task_type", None)
        self.tp_project_name = kwargs.pop("tp_project_name", None)
        self.tp_directory = kwargs.pop("tp_directory", None)
        self.trainer_class_name = kwargs.pop("trainer_class_name", None)