"""Collector that attaches to an ``AgentSession`` and builds a post-call report.

Beta: not covered by semver stability guarantees.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

from ...job import JobContext, get_job_context
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils.misc import is_given
from ...voice.agent_session import AgentSession
from ...voice.events import CloseEvent
from .models import CollectorConfig, PostCallReport


class PostCallTelemetryCollector:
    """Attaches to an ``AgentSession`` and builds a :class:`PostCallReport` on demand.

    Beta: API not covered by semver stability guarantees.

    Does not subscribe to any live per-turn/per-tool events. ``AgentSession`` already
    buffers every event it emits (regardless of whether anything is listening) and
    keeps the running chat history/usage, so :meth:`finalize` does a single
    deterministic reduction over that already-buffered state instead of accumulating
    incremental state via callbacks. This makes attach timing irrelevant to correctness
    — attaching after the session has already started still produces a complete report
    — and makes duplicate/out-of-order event delivery a non-issue.

    Usage::

        collector = PostCallTelemetryCollector(metadata={"lead_id": "lead_123"})
        collector.attach(session)
        await session.start(...)
        ...
        report = collector.finalize()

    Works with or without a ``JobContext`` — ``job_id``/``room_id``/``room_name``/
    ``participant_identity`` are simply ``None`` when unavailable.

    Telemetry and business/CRM data may contain personal or confidential information.
    Enable this collector deliberately, and treat any downstream webhook destination as
    a trusted party.
    """

    def __init__(
        self,
        *,
        config: CollectorConfig | None = None,
        metadata: dict[str, Any] | None = None,
        redact: Callable[[PostCallReport], PostCallReport] | None = None,
    ) -> None:
        """
        Args:
            config: Controls which optional fields (system messages, tool arguments/
                results) are included. Defaults to user+assistant transcript only, with
                tool arguments/results included.
            metadata: Business metadata copied into ``PostCallReport.metadata`` (e.g.
                a CRM contact/lead id). Never place secrets here — it is serialized
                verbatim into the report.
            redact: Optional hook applied to the report immediately before it is cached/
                returned, e.g. to strip PII from the transcript before delivery.
        """
        self._config = config or CollectorConfig()
        self._metadata = dict(metadata) if metadata else {}
        self._redact = redact

        self._session_ref: weakref.ReferenceType[AgentSession] | None = None
        self._job_ctx: NotGivenOr[JobContext | None] = NOT_GIVEN
        self._cached_report: PostCallReport | None = None

    @property
    def attached(self) -> bool:
        return self._session_ref is not None and self._session_ref() is not None

    def attach(
        self, session: AgentSession, *, job_ctx: NotGivenOr[JobContext | None] = NOT_GIVEN
    ) -> None:
        """Attach to ``session``. Safe to call before or after ``session.start()``.

        Re-attaching to the same session is a no-op. Attaching while already attached to
        a different, still-live session raises ``RuntimeError`` — call :meth:`detach`
        first. Attaching to a new session always clears any report cached from a
        previous session, so a reused collector never hands back a stale call's report.

        Args:
            session: The session to observe.
            job_ctx: Left as ``NOT_GIVEN`` (default), the job context is auto-detected
                lazily at :meth:`finalize` time via ``get_job_context(required=False)``.
                Pass an explicit ``JobContext`` to pin it, or ``None`` to force the
                no-job-context path even when one is available (useful for tests).
        """
        current = self._session_ref() if self._session_ref is not None else None
        if current is session:
            return
        if current is not None:
            raise RuntimeError(
                "PostCallTelemetryCollector is already attached to a different "
                "AgentSession; call detach() first"
            )

        self._session_ref = weakref.ref(session)
        self._job_ctx = job_ctx
        self._cached_report = None
        # `.on()`, not `.once()`: EventEmitter.once() registers an internal wrapper
        # closure, not `self._on_close` itself, so a later `.off(event, self._on_close)`
        # in detach() would silently remove nothing and leave the listener firing after
        # detach. `.on()` stores the exact bound method, so `.off()` actually works.
        # Firing more than once is harmless: finalize() is idempotent, and "close" is
        # only ever emitted once per session in any case.
        session.on("close", self._on_close)

    def detach(self) -> None:
        """Stop observing the attached session, if any.

        Does not clear any report already produced by :meth:`finalize` — it remains
        available via a subsequent :meth:`finalize` call until a new session is
        attached.
        """
        session = self._session_ref() if self._session_ref is not None else None
        if session is not None:
            session.off("close", self._on_close)
        self._session_ref = None

    def finalize(self) -> PostCallReport:
        """Build (or return the cached) :class:`PostCallReport`.

        Before the session has closed, this returns a fresh, honestly-partial snapshot
        (``report.ended is False``) on every call — never cached, since a report built
        mid-call would go stale the moment the call actually ends. Once the session has
        closed, the report is built once and cached; every subsequent call returns that
        same, authoritative report.

        Performs no network I/O and never blocks on anything beyond reading in-memory
        session state.
        """
        if self._cached_report is not None and self._cached_report.ended:
            return self._cached_report

        session = self._require_session()
        # snapshot first: _recorded_events is a live, still-growing list while the
        # session runs, and start() swaps in a brand new list on each restart
        events_snapshot = list(session._recorded_events)

        job_ctx = self._resolve_job_ctx()
        job_id = room_id = room_name = participant_identity = None
        if job_ctx is not None:
            job_id = job_ctx.job.id
            room_id = job_ctx.job.room.sid
            room_name = job_ctx.job.room.name
            try:
                participant_identity = job_ctx.local_participant_identity
            except Exception:  # noqa: BLE001 - best-effort field, never fatal
                participant_identity = None

        report = PostCallReport.from_session(
            job_id=job_id,
            room_id=room_id,
            room_name=room_name,
            participant_identity=participant_identity,
            started_at=session._started_at,
            events=events_snapshot,
            chat_history=session.history.copy(),
            model_usage=session.usage.model_usage,
            config=self._config,
            metadata=self._metadata,
        )

        if self._redact is not None:
            report = self._redact(report)

        if report.ended:
            self._cached_report = report

        return report

    def _require_session(self) -> AgentSession:
        if self._session_ref is None:
            raise RuntimeError("PostCallTelemetryCollector.finalize() called before attach()")
        session = self._session_ref()
        if session is None:
            raise RuntimeError(
                "the AgentSession attached to this PostCallTelemetryCollector has been "
                "garbage-collected"
            )
        return session

    def _resolve_job_ctx(self) -> JobContext | None:
        if is_given(self._job_ctx):
            return self._job_ctx
        return get_job_context(required=False)

    def _on_close(self, ev: CloseEvent) -> None:
        # Contained on purpose: a telemetry failure must never break session shutdown.
        # utils.log_exceptions on AgentSession._aclose_impl re-raises after logging, so
        # an uncaught error here would propagate into whatever awaits session.aclose().
        try:
            self.finalize()
        except Exception:
            logger.exception("PostCallTelemetryCollector failed to finalize on session close")
