from fastapi import status

class AutoMLServerError(Exception):
    def __init__(self, msg: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(msg)
        self.status_code = status_code

class ExampleError(AutoMLServerError):
    def __init__(self):
        super().__init__(
            f"ExampleError", status.HTTP_400_BAD_REQUEST
        )