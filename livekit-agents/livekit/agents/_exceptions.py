from __future__ import annotations


class AssignmentTimeoutError(Exception):
    """Raised when accepting a job but not receiving an assignment within the specified timeout.
    The server may have chosen another worker to handle this job."""

    pass


# errors used by our plugins


class APIError(Exception):
    """Raised when an API request failed.
    This is used on our TTS/STT/LLM plugins."""

    message: str
    """
    The error message returned by the API.
    """

    body: object | None
    """The API response body, if available.


    If the API returned a valid json, the body will contains
    the decodede result.
    """

    retryable: bool = False
    """Whether the error can be retried."""

    def __init__(
        self, message: str, *, body: object | None, retryable: bool = True
    ) -> None:
        super().__init__(message)

        self.message = message
        self.body = body
        self.retryable = retryable


class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""

    status_code: int
    """The status code of the API response."""

    request_id: str | None
    """The request ID of the API response, if available."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = -1,
        request_id: str | None = None,
        body: object | None = None,
        retryable: bool | None = None,
    ) -> None:
        if retryable is None:
            retryable = True
            # 4xx errors are not retryable
            if status_code >= 400 and status_code < 500:
                retryable = False

        super().__init__(message, body=body, retryable=retryable)

        self.status_code = status_code
        self.request_id = request_id


class APIConnectionError(APIError):
    """Raised when an API request failed due to a connection error."""

    def __init__(
        self, message: str = "Connection error.", *, retryable: bool = True
    ) -> None:
        super().__init__(message, body=None, retryable=retryable)


class APITimeoutError(APIConnectionError):
    """Raised when an API request timed out."""

    def __init__(
        self, message: str = "Request timed out.", *, retryable: bool = True
    ) -> None:
        super().__init__(message, retryable=retryable)
