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

class DeleteExperimentJobError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class CreateExperimentJobError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class GetExperimentJobLogsError(AutoMLServerError):
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

class GetExperimentJobStatusError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class ExperimentNotExistError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class ExperimentNameError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class ParseExperimentSummaryError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class GetSessionError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class WebSocketQueryParamError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code

class ValueError(AutoMLServerError):
    def __init__(self, msg: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(msg)
        self.status_code = status_code