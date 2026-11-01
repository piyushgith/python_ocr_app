class AppError(Exception):
    """Error that is safe to show to API clients."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def unsupported_format(message: str) -> AppError:
    return AppError("UNSUPPORTED_FORMAT", message, 415)


def corrupted_document(message: str) -> AppError:
    return AppError("CORRUPTED_DOCUMENT", message, 422)
