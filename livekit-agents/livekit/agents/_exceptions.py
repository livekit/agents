from __future__ import annotations


class UnexpectedModelBehavior(RuntimeError):
    """Raised when the model behaves in a way the run cannot recover from,
    e.g. a run with an output_type ends without the expected output after
    exhausting its retries."""


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

        # 4xx client errors are not retryable, except for transient codes:
        # 408 (Request Timeout), 429 (Too Many Requests), 499 (Client Closed Request / gRPC CANCELLED).
        # This overrides any caller-provided retryable=True, since a client
        # error (bad URL, auth, bad request) will keep failing on retry.
        if 400 <= status_code < 500 and status_code not in (408, 429, 499):
            retryable = False

        super().__init__(message, body=body, retryable=retryable)

        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [
            f"message={self.message!r}",
            f"status_code={self.status_code}",
            f"retryable={self.retryable}",
        ]
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        if self.body:
            parts.append(f"body={self.body}")
        return ", ".join(parts)

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

    def __str__(self) -> str:
        # `raise ... from e` sets __cause__ after __init__, so the chain is unreachable at
        # construction and must be read here. The default message carries no detail, and
        # providers wrap transport failures behind placeholder messages of their own, so the
        # root of the chain is the only place the actual failure is named.
        # a chain may cycle back on itself, so the walk stops at the first repeated exception
        # and after 10 distinct ones.
        root: BaseException | None = None
        seen = {id(self)}
        cause = self.__cause__
        for _ in range(10):
            if cause is None or id(cause) in seen:
                break
            seen.add(id(cause))
            root = cause
            cause = cause.__cause__

        if root is None:
            return self.message

        # naming our own type by .message keeps a cycle that ends on one out of __str__, which
        # would otherwise re-enter here through the same chain.
        if isinstance(root, APIConnectionError):
            detail = f"{type(root).__name__}: {root.message}"
        else:
            # a third-party __str__ may raise (e.g. aiohttp.ClientConnectorError on a partially
            # initialized connection key); the type name alone still beats losing the message.
            try:
                detail = f"{type(root).__name__}: {root}"
            except Exception:
                detail = type(root).__name__

        return f"{self.message} (caused by {detail})"


class APITimeoutError(APIConnectionError):
    """Raised when an API request timed out."""

    def __init__(self, message: str = "Request timed out.", *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class CLIError(Exception):
    pass


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
