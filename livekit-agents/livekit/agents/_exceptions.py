from __future__ import annotations

INFERENCE_QUOTA_EXCEEDED_TYPE = "inference_quota_exceeded"
"""Value of the ``type`` field in a LiveKit Inference 429 quota response body."""

# The gateway returns `inference_quota_exceeded` for two different classes of 429.
# These categories mean a billing quota is exhausted ("Wait for the next billing
# cycle …") — they will fail identically every turn until the quota resets, so they
# are terminal and non-retryable. Every other category (rate/concurrency limits like
# MaxConcurrentGatewayLLMRpm/Tpm) is transient: it recovers via backoff, so it stays
# retryable and non-terminal. See agent-gateway `pkg/quota/response.go::quotaHint`.
_TERMINAL_QUOTA_CATEGORIES = frozenset({"MaxGatewayCredits", "MaxBargeInRequests"})


def _str_or_none(value: object) -> str | None:
    """Coerce an untrusted JSON field to ``str``; non-str values become ``None``."""
    return value if isinstance(value, str) else None


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
    """Whether the error can be retried (within the request's retry loop)."""

    terminal: bool = False
    """Whether the error is terminal — it will fail identically on every turn, so
    callers should surface it immediately rather than absorbing it under a
    transient-error tolerance (e.g. ``AgentSession``'s ``max_unrecoverable_errors``).

    Independent of ``retryable``: ``retryable`` governs in-request retries, while
    ``terminal`` governs whether higher-level loops should give up at once. A quota
    error from depleted credits is both non-retryable and terminal; a transient
    rate-limit is non-terminal (and may be retryable)."""

    def __init__(
        self,
        message: str,
        *,
        body: object | None = None,
        retryable: bool = True,
        terminal: bool = False,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.body = body
        self.retryable = retryable
        self.terminal = terminal

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
        terminal: bool = False,
    ) -> None:
        if retryable is None:
            retryable = True

        # 4xx client errors are not retryable, except for transient codes:
        # 408 (Request Timeout), 429 (Too Many Requests), 499 (Client Closed Request / gRPC CANCELLED).
        # This overrides any caller-provided retryable=True, since a client
        # error (bad URL, auth, bad request) will keep failing on retry.
        if 400 <= status_code < 500 and status_code not in (408, 429, 499):
            retryable = False

        super().__init__(message, body=body, retryable=retryable, terminal=terminal)

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


class APIQuotaExceededError(APIStatusError):
    """Raised when the inference gateway rejects a request because a usage quota
    or rate limit has been exhausted.

    LiveKit Inference answers an exhausted project with ``HTTP 429`` and a
    structured JSON body (``type == "inference_quota_exceeded"``). This error
    surfaces the fields of that body directly so callers can render or speak a
    precise, user-facing message (``hint``) instead of leaving the agent silent.

    The gateway uses this single ``type`` for two different conditions, told apart by
    ``category``:

    * **Credit/quota exhaustion** (``MaxGatewayCredits``, ``MaxBargeInRequests``) —
      recovers only at the next billing cycle, so it is :attr:`terminal` and
      ``retryable=False``.
    * **Rate / concurrency limits** (e.g. ``MaxConcurrentGatewayLLMRpm`` / ``…Tpm``) —
      recover within ~a minute via backoff, so they stay ``retryable=True`` and
      non-terminal (they fall through the usual transient-error handling).

    ``retryable`` / ``terminal`` are derived from ``category`` automatically; pass them
    explicitly to override.

    On a terminal quota error, ``AgentSession`` by default speaks the ``hint`` and
    closes on the first occurrence (see ``AgentSession(error_message=...)``); transient
    variants go through the normal retry/tolerance path. Subscribe to ``error`` only
    when you need the structured fields, e.g. to forward an "out of credits" state to
    your frontend. ``ErrorEvent.error`` is the ``LLMError``/``STTError``/… wrapper, so
    the underlying exception is at ``ev.error.error``:

    Example:
        ```python
        from livekit.agents import APIQuotaExceededError, ErrorEvent


        @session.on("error")
        def _on_error(ev: ErrorEvent) -> None:
            err = ev.error.error
            if isinstance(err, APIQuotaExceededError):
                logger.warning("inference quota exceeded: %s (%s)", err.hint, err.quota_type)
        ```
    """

    quota_type: str | None
    """Which resource ran out, e.g. ``"llm"``, ``"stt"``, ``"tts"`` or ``"bargein"``."""

    category: str | None
    """Gateway category. Credit-exhaustion categories (``"MaxGatewayCredits"``,
    ``"MaxBargeInRequests"``) are terminal; rate-limit variants such as
    ``"MaxConcurrentGatewayLLMRpm"`` are transient."""

    hint: str | None
    """Human-readable, user-appropriate explanation suitable to speak or display."""

    remaining_limit: str | None
    """Remaining quota for ``quota_type`` as reported by the gateway; ``"0"`` when
    fully exhausted. An opaque diagnostic string (not guaranteed numeric)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 429,
        request_id: str | None = None,
        body: object | None = None,
        retryable: bool | None = None,
        terminal: bool | None = None,
        quota_type: str | None = None,
        category: str | None = None,
        hint: str | None = None,
        remaining_limit: str | None = None,
    ) -> None:
        # the response body carries the structured fields; read category early so we
        # can derive retryable/terminal from it when not given explicitly. The body is
        # wire data from a user-configurable endpoint, so non-str values are dropped —
        # they'd violate the `str | None` fields and break the category check below.
        if isinstance(body, dict):
            if quota_type is None:
                quota_type = _str_or_none(body.get("quota_type"))
            if category is None:
                category = _str_or_none(body.get("category"))
            if hint is None:
                hint = _str_or_none(body.get("hint"))
            if remaining_limit is None:
                remaining_limit = _str_or_none(body.get("remaining_limit"))

        # credit exhaustion is terminal and won't recover on retry; everything else
        # (rate/concurrency limits, or an unknown category) is treated as transient
        is_credit_exhaustion = category in _TERMINAL_QUOTA_CATEGORIES
        if terminal is None:
            terminal = is_credit_exhaustion
        if retryable is None:
            retryable = not is_credit_exhaustion

        super().__init__(
            message,
            status_code=status_code,
            request_id=request_id,
            body=body,
            retryable=retryable,
            terminal=terminal,
        )

        self.quota_type = quota_type
        self.category = category
        self.hint = hint
        self.remaining_limit = remaining_limit

    @classmethod
    def from_response(
        cls,
        message: str,
        *,
        status_code: int = 429,
        request_id: str | None = None,
        body: object | None = None,
    ) -> APIQuotaExceededError | None:
        """Build an :class:`APIQuotaExceededError` from a response body, or return
        ``None`` if the body isn't a LiveKit Inference quota-exceeded payload.

        Lets plugins centralize quota detection: pass the decoded JSON body and
        raise the result when it isn't ``None``.
        """
        if not (isinstance(body, dict) and body.get("type") == INFERENCE_QUOTA_EXCEEDED_TYPE):
            return None

        return cls(message, status_code=status_code, request_id=request_id, body=body)


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

    quota_error = APIQuotaExceededError.from_response(
        display, status_code=status, request_id=request_id, body=body
    )
    if quota_error is not None:
        return quota_error

    return APIStatusError(
        message=display,
        status_code=status,
        request_id=request_id,
        body=body,
    )
