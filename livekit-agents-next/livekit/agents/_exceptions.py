from __future__ import annotations

from ._generated.api_errors import (
    APIConnectionError as APIConnectionError,
    APIError as APIError,
    APIStatusError as APIStatusError,
    APITimeoutError as APITimeoutError,
    AssignmentTimeoutError as AssignmentTimeoutError,
    CLIError as CLIError,
    RedactedCause as RedactedCause,
    StreamStateError as StreamStateError,
    UnexpectedModelBehavior as UnexpectedModelBehavior,
)


def _status_retryable(status_code: int, retryable: bool | None) -> bool:
    # Preserve the public constructor's transient 499 behavior. Native cancellation has its
    # own policy and remains non-retryable.
    if 400 <= status_code < 500 and status_code not in (408, 429, 499):
        return False
    return True if retryable is None else retryable


def _status_error_str(self: APIStatusError) -> str:
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


def _connection_error_str(self: APIConnectionError) -> str:
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
        root_message = root.message
    else:
        # a third-party __str__ may raise (e.g. aiohttp.ClientConnectorError on a partially
        # initialized connection key); the type name alone still beats losing the message.
        try:
            root_message = str(root)
        except Exception:
            root_message = ""

    detail = type(root).__name__
    if root_message:
        detail += f": {root_message}"

    return f"{self.message} (caused by {detail})"


def redacted_cause(error: BaseException) -> RedactedCause:
    """``error`` to chain as a cause (``raise APIError(...) from redacted_cause(e)``)."""
    return RedactedCause(type(error).__qualname__).with_traceback(error.__traceback__)


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

    from . import _core

    reason = HTTPStatus(status).phrase if status in HTTPStatus._value2member_map_ else None
    return APIStatusError(
        message=_core.http_error_display(message, status, reason),
        status_code=status,
        request_id=request_id,
        body=body,
    )
