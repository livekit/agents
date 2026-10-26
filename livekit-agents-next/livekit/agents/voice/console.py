"""The ``console`` subcommand's call, as the session runs it: ``lk agent console`` owns the
microphone, the speakers and the TUI, and reaches the agent over one socket."""

from __future__ import annotations

import datetime
import json
import pathlib
import time
from typing import TYPE_CHECKING, Any, ClassVar

from .. import _core, llm
from .._core_view import to_core_items
from ..job import get_job_context
from ..llm.chat_context import Instructions
from ..log import logger
from ..version import __version__

if TYPE_CHECKING:
    from .agent_session import LiveSession
    from .events import FunctionToolsExecutedEvent
    from .run_result import RunResult

# the core's session events the CLI renders; conversation items are filtered further
_CONSOLE_EVENTS = {
    "agent_state_changed",
    "user_state_changed",
    "user_input_transcribed",
    "conversation_item_added",
    "tool_execution_updated",
    "eot_prediction",
    "agent_false_interruption",
    "error",
}


class ConsoleCall:
    """The ``console`` subcommand's call: the CLI's socket (``_core.ConsoleIo``) is the session's
    microphone and speaker, and a ``_core.RemoteSessionHost`` on it forwards the session's events
    to the CLI and answers its requests (typed turns, muting) through this object."""

    addr: ClassVar[str | None] = None
    """The CLI's ``host:port`` while the subcommand runs; the first session started takes it."""
    record: ClassVar[bool] = False

    def __init__(self, session: LiveSession[Any], addr: str) -> None:
        self.io = _core.ConsoleIo(addr)
        self._session = session
        self._host = _core.RemoteSessionHost()
        self._host.set_port(self)
        # answered through a hook, so not among the core's events
        session.on("function_tools_executed", self._tools_executed)

    @classmethod
    def claim(cls, session: LiveSession[Any]) -> ConsoleCall | None:
        if cls.addr is None:
            return None
        addr, cls.addr = cls.addr, None
        return cls(session, addr)

    async def start(self) -> None:
        await self._host.start(self.io, self.io)
        # the call ends with the job: Ctrl+C in the CLI
        get_job_context().add_shutdown_callback(self._session.aclose)

    def forward(self, wire: dict[str, Any]) -> None:
        """Send one of the core's session events to the CLI."""
        kind = wire["type"]
        if kind not in _CONSOLE_EVENTS:
            return
        if kind == "conversation_item_added":
            if wire["item"].type not in ("message", "agent_handoff"):
                return
        elif kind == "error":
            wire = {"message": str(wire["error"])}
        elif kind == "tool_execution_updated" and "function_call" in wire["update"]:
            call = wire["update"]["function_call"].fields()
            keys = ("id", "call_id", "name", "arguments")
            wire = {"update": {"type": "tool_call_started", **{k: call[k] for k in keys}}}
        self._send({**wire, "kind": kind})

    def _tools_executed(self, ev: FunctionToolsExecutedEvent) -> None:
        self._send(
            {
                "kind": "function_tools_executed",
                "function_calls": [(c.name, c.arguments, c.call_id) for c in ev.function_calls],
                "function_call_outputs": [
                    (o.call_id, o.name, o.output, o.is_error)
                    for o in ev.function_call_outputs
                    if o is not None
                ],
            }
        )

    def _send(self, event: dict[str, Any]) -> None:
        try:
            self._host.send_event(event)
        except Exception:
            logger.exception("failed to forward a session event to the console")

    async def aclose(self) -> None:
        await self._host.aclose(2.0)
        if not ConsoleCall.record:
            return
        directory = pathlib.Path(
            "console-recordings", f"session-{datetime.datetime.now().strftime('%m-%d-%H%M%S')}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        report = get_job_context().make_session_report(self._session)
        (directory / "session_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    # --- what the CLI asks of the session (`RemoteSessionHost`'s port) ---------------------

    async def _core_chat_history(self) -> list[_core.ChatItem]:
        return to_core_items(self._session.history.items)

    async def _core_agent_info(self) -> tuple[str, str | None, list[str], list[_core.ChatItem]]:
        agent = self._session.current_agent
        instructions = agent.instructions
        rendered = (
            instructions.render(modality="audio")
            if isinstance(instructions, Instructions)
            else str(instructions)
        )
        tools = list(llm.ToolContext(agent.tools).function_tools)
        return agent.id, rendered, tools, to_core_items(self._session.history.items)

    async def _core_run_input(self, text: str) -> tuple[list[_core.ChatItem] | None, str | None]:
        if not text:
            return None, "empty run_input text"
        try:
            await self._session.interrupt(force=True)
            result: RunResult[None] = self._session.run(user_input=text)
            await result
        except Exception as e:
            return None, str(e)
        return to_core_items([ev.item for ev in result.events]), None

    async def _core_session_state(self) -> tuple[str, str, str, dict[str, str], float]:
        options = self._session.options
        return (
            self._session.agent_state,
            self._session.user_state,
            self._session.current_agent.id,
            {k: str(options.get(k)) for k, _ in _core.session_state_option_fields()},
            self._session._started_at or time.time(),
        )

    async def _core_rtc_stats(self) -> None:
        return None

    async def _core_session_usage(self) -> list[Any]:
        return list(self._session.usage.model_usage)

    def _core_framework_info(self) -> tuple[str, str]:
        return "python", __version__

    async def _core_update_io(self, update: tuple[bool | None, ...]) -> None:
        audio_in, _, audio_out, _, transcription = update
        if audio_in is not None:
            self._session.input.set_audio_enabled(audio_in)
        if audio_out is not None:
            self._session.output.set_audio_enabled(audio_out)
        if transcription is not None:
            self._session.output.set_transcription_enabled(transcription)

    async def _core_finalize_simulation(self, provisional: tuple[bool, str]) -> tuple[None, None]:
        return None, None
