from __future__ import annotations

from .._exceptions import APIStatusError

_INFERENCE_QUOTA_EXCEEDED_TYPE = "inference_quota_exceeded"
"""Value of the ``type`` field in a LiveKit Inference 429 quota response body."""

_TERMINAL_QUOTA_CATEGORIES = frozenset(
    {"MaxGatewayCredits", "MaxBargeInRequests", "MaxEotRequests"}
)


def _str_or_none(value: object) -> str | None:
    """Coerce an untrusted JSON field to ``str``; non-str values become ``None``."""
    return value if isinstance(value, str) else None


class InferenceQuotaExceededError(APIStatusError):
    """Raised when the LiveKit Inference gateway rejects a request because a usage quota
    or rate limit has been exhausted.

    The gateway answers an exhausted project with ``HTTP 429`` and a structured JSON
    body (``type == "inference_quota_exceeded"``). This error surfaces the fields of that
    body directly so callers can log the quota state or forward it to their frontend
    instead of leaving the agent silent.

    The gateway uses this single ``type`` for two different conditions, told apart by
    ``category``:

    * **Credit/quota exhaustion** (``MaxGatewayCredits``, ``MaxBargeInRequests``,
      ``MaxEotRequests``) — recovers only at the next billing cycle, so it is
      :attr:`terminal` and ``retryable=False``.
    * **Rate / concurrency limits** (e.g. ``MaxConcurrentGatewayLLMRpm`` / ``…Tpm``) —
      recover within ~a minute via backoff, so they stay ``retryable=True`` and
      non-terminal (they fall through the usual transient-error handling).

    ``retryable`` / ``terminal`` are derived from ``category`` automatically; pass them
    explicitly to override.

    On a terminal quota error, ``AgentSession`` by default speaks a generic,
    provider-agnostic message and closes on the first occurrence (see
    ``AgentSession(unrecoverable_error_message=...)``); transient variants go through the
    normal retry/tolerance path. The gateway ``hint`` is never spoken — quota details
    aren't surfaced to end users. Subscribe to ``error`` when you need the structured
    fields, e.g. to log the quota state or forward an "out of credits" state to your
    frontend. ``ErrorEvent.error`` is the ``LLMError``/``STTError``/… wrapper, so the
    underlying exception is at ``ev.error.error``:

    Example:
        ```python
        from livekit.agents import ErrorEvent
        from livekit.agents.inference import InferenceQuotaExceededError


        @session.on("error")
        def _on_error(ev: ErrorEvent) -> None:
            err = ev.error.error
            if isinstance(err, InferenceQuotaExceededError):
                logger.warning("inference quota exceeded: %s (%s)", err.hint, err.quota_type)
        ```
    """

    quota_type: str | None
    """Quota resource: ``"llm"``, ``"stt"``, ``"tts"``, ``"bargein"`` or ``"eot"``."""

    category: str | None
    """Quota error category. Credit-exhaustion categories (``"MaxGatewayCredits"``,
    ``"MaxBargeInRequests"``, ``"MaxEotRequests"``) are terminal; rate-limit variants
    such as ``"MaxConcurrentGatewayLLMRpm"`` are transient."""

    hint: str | None
    """Human-readable explanation from the error."""

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
        if isinstance(body, dict):
            if quota_type is None:
                quota_type = _str_or_none(body.get("quota_type"))
            if category is None:
                category = _str_or_none(body.get("category"))
            if hint is None:
                hint = _str_or_none(body.get("hint"))
            if remaining_limit is None:
                remaining_limit = _str_or_none(body.get("remaining_limit"))

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
    ) -> InferenceQuotaExceededError | None:
        """Build an :class:`InferenceQuotaExceededError` from a response body, or return
        ``None`` if the body isn't a LiveKit Inference quota-exceeded payload.

        Lets plugins centralize quota detection: pass the decoded JSON body and
        raise the result when it isn't ``None``.
        """
        if not (isinstance(body, dict) and body.get("type") == _INFERENCE_QUOTA_EXCEEDED_TYPE):
            return None

        return cls(message, status_code=status_code, request_id=request_id, body=body)
