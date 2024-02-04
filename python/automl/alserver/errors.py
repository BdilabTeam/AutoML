from fastapi import status

class AutoMLServerError(Exception):
    def __init__(self, msg: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(msg)
        self.status_code = status_code

class DataFormatError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(msg)
        self.status_code = status_code

class MySQLNotExistError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class SelectModelError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class DeleteJobError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class CreateJobError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class GetJobInfoError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class GetTrainingParamsError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class SaveTrainingParamsError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class TrainingProjectNotExistError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code