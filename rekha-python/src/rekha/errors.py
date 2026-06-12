class RekhaError(Exception):
    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(f"{message}" + (f": {detail}" if detail else ""))


class RekhaConnectError(RekhaError):
    def __init__(self, seed: str, detail: str):
        super().__init__(f"failed to connect to {seed}", detail)


class RekhaRequestError(RekhaError):
    def __init__(self, operation: str, status_code: str, detail: str):
        self.operation = operation
        self.status_code = status_code
        super().__init__(f"{operation} failed ({status_code})", detail)
