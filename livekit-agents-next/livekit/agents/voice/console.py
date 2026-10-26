"""The ``console`` subcommand's call, as the session runs it: ``lk agent console`` owns the
microphone, the speakers and the TUI, and reaches the agent over one socket."""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

from .. import _core
from ..job import get_job_context
from .remote_session import SessionHost

if TYPE_CHECKING:
    from .agent_session import LiveSession


class ConsoleCall(SessionHost):
    """The ``console`` subcommand's call: the CLI's socket (``_core.ConsoleIo``) is the session's
    microphone and speaker, and the session's host on it forwards the session's events to the CLI
    and answers its requests (typed turns, muting)."""

    # the core's session events the CLI renders, and the conversation items among them
    EVENTS = SessionHost.EVENTS | {"eot_prediction"}
    ITEMS = frozenset({"message", "agent_handoff"})
    DRAIN = 2.0

    addr: ClassVar[str | None] = None
    """The CLI's ``host:port`` while the subcommand runs; the first session started takes it."""
    record: ClassVar[bool] = False

    def __init__(self, session: LiveSession[Any], addr: str) -> None:
        self.io = _core.ConsoleIo(addr)
        super().__init__(session, self.io, self.io)

    @classmethod
    def claim(cls, session: LiveSession[Any]) -> ConsoleCall | None:
        if cls.addr is None:
            return None
        addr, cls.addr = cls.addr, None
        return cls(session, addr)

    async def start(self) -> None:
        await super().start()
        # the call ends with the job: Ctrl+C in the CLI
        get_job_context().add_shutdown_callback(self._session.aclose)

    async def aclose(self) -> None:
        await super().aclose()
        if not ConsoleCall.record:
            return
        directory = pathlib.Path(
            "console-recordings", f"session-{datetime.datetime.now().strftime('%m-%d-%H%M%S')}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        report = get_job_context().make_session_report(self._session)
        (directory / "session_report.json").write_text(json.dumps(report.to_dict(), indent=2))
