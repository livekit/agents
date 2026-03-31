class OjinException(Exception):
    """Exception for Ojin avatar errors."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        code: str | None = None,
        origin: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = code
        self.origin = origin
