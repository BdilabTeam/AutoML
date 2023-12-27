from fastapi import status

from ..errors import AutoMLServerError


class CreateTFJobError(AutoMLServerError):
    def __init__(self, msg: str):
        super().__init__(
            msg=msg, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class GetTFJobError(AutoMLServerError):
    def __init__(self, msg: str):
        super().__init__(
            msg=msg, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
class GetTFJobConditionsError(AutoMLServerError):
    def __init__(self, msg: str):
        super().__init__(
            msg=msg, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class GetTFJobLogsError(AutoMLServerError):
    def __init__(self, msg: str):
        super().__init__(
            msg=msg, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class ValueError(AutoMLServerError):
    def __init__(self, msg: str):
        super().__init__(
            msg=msg, 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )