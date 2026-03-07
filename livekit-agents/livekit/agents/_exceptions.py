from __future__ import annotations

from livekit.protocol.agent_pb import agent_text


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

    def __init__(self, message: str, *, body: object | None = None, retryable: bool = True) -> None:
        super().__init__(message)

        self.message = message
        self.body = body
        self.retryable = retryable

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, body={self.body!r}, retryable={self.retryable!r})"


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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.message!r}, "
            f"status_code={self.status_code!r}, "
            f"request_id={self.request_id!r}, "
            f"body={self.body!r}, "
            f"retryable={self.retryable!r})"
        )


class APIConnectionError(APIError):
    """Raised when an API request failed due to a connection error."""

    def __init__(self, message: str = "Connection error.", *, retryable: bool = True) -> None:
        super().__init__(message, body=None, retryable=retryable)


class APITimeoutError(APIConnectionError):
    """Raised when an API request timed out."""

    def __init__(self, message: str = "Request timed out.", *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class CLIError(Exception):
    pass


class TextMessageError(Exception):
    def __init__(
        self,
        message: str,
        code: agent_text.TextMessageErrorCode = agent_text.INTERNAL_ERROR,
    ) -> None:
        super().__init__(message)
        self._message = message
        self._code = code

    @property
    def message(self) -> str:
        return self._message

    @property
    def code(self) -> agent_text.TextMessageErrorCode:
        return self._code

    def to_proto(self) -> agent_text.TextMessageError:
        return agent_text.TextMessageError(message=self._message, code=self._code)


def create_api_error_from_http(
    message: str = "",
    *,
    status: int,
    request_id: str | None = None,
    body: object | None = None,
) -> APIStatusError:
    """Create an APIStatusError from an HTTP status code.

    When the message carries extra detail beyond the standard reason phrase,
    both the message and the reason are shown. Otherwise just the reason.
    """
    from http import HTTPStatus

    try:
        reason = HTTPStatus(status).phrase
    except ValueError:
        reason = f"HTTP {status}"

    if message and message != reason:
        display = f"{message} ({status} {reason})"
    else:
        display = f"{reason} ({status})"

    return APIStatusError(
        message=display,
        status_code=status,
        request_id=request_id,
        body=body,
    )
