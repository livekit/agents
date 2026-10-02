from __future__ import annotations

import contextvars
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .._exceptions import APIError, APIStatusError
from ..utils import shortuuid

ProviderRequestComponent = Literal["stt", "tts", "llm", "realtime"]
ProviderRequestOutcome = Literal["success", "error", "cancelled"]
ProviderRequestPurpose = Literal["foreground", "recovery"]


class ProviderRequestAttempt(BaseModel):
    """Content-free metadata for one request attempt against a model provider."""

    model_config = ConfigDict(frozen=True)

    component: ProviderRequestComponent
    provider: str | None = None
    model: str | None = None
    operation_id: str
    """SDK-local ID shared by retries and fallback attempts for one logical operation."""
    sdk_request_id: str
    """SDK-local ID for this specific attempt. It is never a provider correlation ID."""
    provider_request_ids: tuple[str, ...] = Field(default_factory=tuple)
    provider_trace_ids: tuple[str, ...] = Field(default_factory=tuple)
    speech_id: str | None = None
    purpose: ProviderRequestPurpose = "foreground"
    """Whether this attempt serves a user operation or probes provider recovery."""
    started_at: float
    completed_at: float
    outcome: ProviderRequestOutcome
    retry_index: int = 0
    fallback_index: int = 0
    error_type: str | None = None
    status_code: int | None = None
    retryable: bool | None = None


class ProviderRequestLedger:
    """A bounded, in-process history of provider request attempts.

    The oldest attempt is deterministically evicted when ``capacity`` is reached.
    Snapshots are immutable tuples and attempts contain operational metadata only.
    """

    def __init__(self, *, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        self._capacity = capacity
        self._attempts: deque[ProviderRequestAttempt] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def record(self, attempt: ProviderRequestAttempt) -> None:
        """Append an attempt, evicting the oldest entry when full."""
        with self._lock:
            self._attempts.append(attempt)

    def snapshot(self) -> tuple[ProviderRequestAttempt, ...]:
        """Return an immutable point-in-time copy, oldest attempt first."""
        with self._lock:
            return tuple(self._attempts)

    def __len__(self) -> int:
        with self._lock:
            return len(self._attempts)


@dataclass(frozen=True)
class _ProviderRequestContext:
    operation_id: str
    fallback_index: int
    purpose: ProviderRequestPurpose = "foreground"


_context_var = contextvars.ContextVar[_ProviderRequestContext | None](
    "provider_request_context", default=None
)


@contextmanager
def _provider_request_context(
    operation_id: str,
    fallback_index: int,
    purpose: ProviderRequestPurpose = "foreground",
) -> Iterator[None]:
    token = _context_var.set(
        _ProviderRequestContext(
            operation_id=operation_id,
            fallback_index=fallback_index,
            purpose=purpose,
        )
    )
    try:
        yield
    finally:
        _context_var.reset(token)


@contextmanager
def _provider_request_fallback_context(fallback_index: int) -> Iterator[None]:
    current = _context_var.get()
    operation_id = current.operation_id if current else shortuuid("op_")
    purpose = current.purpose if current else "foreground"
    with _provider_request_context(operation_id, fallback_index, purpose):
        yield


@contextmanager
def _provider_request_recovery_context(fallback_index: int) -> Iterator[None]:
    """Create an independent operation for a background provider recovery probe."""
    with _provider_request_context(shortuuid("op_"), fallback_index, "recovery"):
        yield


class _ProviderRequestTracker:
    """Mutable per-stream helper that emits immutable completed attempts."""

    def __init__(
        self,
        *,
        component: ProviderRequestComponent,
        provider: str | None,
        model: str | None,
        operation_id: str | None = None,
    ) -> None:
        context = _context_var.get()
        self.component = component
        self.provider = None if provider in (None, "unknown") else provider
        self.model = None if model in (None, "unknown") else model
        self.operation_id = operation_id or (context.operation_id if context else shortuuid("op_"))
        self.fallback_index = context.fallback_index if context else 0
        self.purpose: ProviderRequestPurpose = context.purpose if context else "foreground"
        self.sdk_request_id = ""
        self.retry_index = 0
        self.started_at = 0.0
        self.provider_request_ids: list[str] = []
        self.provider_trace_ids: list[str] = []

    def start(
        self,
        retry_index: int,
        *,
        sdk_request_id: str | None = None,
        started_at: float | None = None,
    ) -> None:
        self.sdk_request_id = sdk_request_id or shortuuid("req_")
        self.retry_index = retry_index
        self.started_at = started_at if started_at is not None else time.time()
        self.provider_request_ids = []
        self.provider_trace_ids = []

    def note_provider_request_id(self, request_id: str | None) -> None:
        if request_id and request_id not in self.provider_request_ids:
            self.provider_request_ids.append(request_id)

    def note_provider_trace_id(self, trace_id: str | None) -> None:
        if trace_id and trace_id not in self.provider_trace_ids:
            self.provider_trace_ids.append(trace_id)

    def complete(
        self,
        outcome: ProviderRequestOutcome,
        *,
        error: BaseException | None = None,
        completed_at: float | None = None,
    ) -> ProviderRequestAttempt:
        if isinstance(error, APIStatusError):
            self.note_provider_request_id(error.request_id)
        return ProviderRequestAttempt(
            component=self.component,
            provider=self.provider,
            model=self.model,
            operation_id=self.operation_id,
            sdk_request_id=self.sdk_request_id,
            provider_request_ids=tuple(self.provider_request_ids),
            provider_trace_ids=tuple(self.provider_trace_ids),
            purpose=self.purpose,
            started_at=self.started_at,
            completed_at=completed_at if completed_at is not None else time.time(),
            outcome=outcome,
            retry_index=self.retry_index,
            fallback_index=self.fallback_index,
            error_type=type(error).__name__ if error is not None else None,
            status_code=error.status_code if isinstance(error, APIStatusError) else None,
            retryable=error.retryable if isinstance(error, APIError) else None,
        )
