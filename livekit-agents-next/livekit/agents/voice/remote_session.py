"""The remote session protocol a client drives a call through: ``lk agent simulate`` and the
dashboard over the room's ``lk.agent.session`` byte streams, ``lk agent console`` over its socket.
The core serves it for the session (``VoiceSession(remote_session=)``); this module answers what a
client asks that needs the session's objects, and holds the console subcommand's socket."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from google.protobuf.json_format import MessageToDict

from .. import _core, llm
from .._core_view import to_core_items
from ..job import get_job_context
from ..llm.chat_context import Instructions
from ..log import logger
from ..version import __version__

if TYPE_CHECKING:
    from .agent_session import VoiceSession
    from .run_result import VoiceRunResult


class ConsoleCall:
    """The ``console`` subcommand's call: ``lk agent console`` owns the microphone, the speakers
    and the TUI, and reaches the agent over one socket, which the first session started takes as
    its audio and its remote session's transport."""

    addr: ClassVar[str | None] = None
    """The CLI's ``host:port`` while the subcommand runs."""
    record: ClassVar[bool] = False

    @classmethod
    def claim(cls) -> _core.ConsoleIo | None:
        if cls.addr is None:
            return None
        addr, cls.addr = cls.addr, None
        return _core.ConsoleIo(addr)


class SessionPort:
    """What a remote client asks of ``session`` that needs its objects (``_core.SessionPort``)."""

    def __init__(self, session: VoiceSession[Any]) -> None:
        self._session = session

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
        return agent.name, rendered, tools, to_core_items(self._session.history.items)

    async def _core_run_input(self, text: str) -> tuple[list[_core.ChatItem] | None, str | None]:
        try:
            await self._session.interrupt(force=True)
            result: VoiceRunResult[None] = self._session.run(user_input=text)
            await result
        except Exception as e:
            return None, str(e)
        return to_core_items([ev.item for ev in result.events]), None

    async def _core_session_state(self) -> tuple[str, str, str, dict[str, str], float | None]:
        options = self._session.options
        return (
            self._session.agent_state,
            self._session.user_state,
            self._session.current_agent.name,
            {k: str(options.get(k)) for k, _ in _core.session_state_option_fields()},
            self._session._started_at,
        )

    async def _core_rtc_stats(self) -> tuple[list[Any], list[Any]] | None:
        job_ctx = get_job_context(required=False)
        if job_ctx is None or not job_ctx.room.isconnected():
            return None
        stats = await job_ctx.room.get_rtc_stats()
        return (
            [MessageToDict(s) for s in stats.publisher_stats],
            [MessageToDict(s) for s in stats.subscriber_stats],
        )

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

    async def _core_finalize_simulation(
        self, provisional: tuple[bool, str]
    ) -> tuple[tuple[bool, str] | None, str | None]:

        from ..simulation import SimulationVerdict

        job_ctx = get_job_context(required=False)
        sim_ctx = job_ctx.simulation_context() if job_ctx is not None else None
        if job_ctx is None or sim_ctx is None:
            return None, None
        try:
            success, reason = provisional
            sim_ctx._begin_finalize(
                simulator_verdict=SimulationVerdict(success=success, reason=reason),
            )
            if (fnc := job_ctx._simulation_end_fnc) is not None:
                if asyncio.iscoroutine(out := fnc(sim_ctx)):
                    await out
        except Exception as e:
            logger.exception("error while executing the on_simulation_end callback")
            return None, str(e)
        verdict = sim_ctx.user_verdict
        return ((verdict.success, verdict.reason) if verdict else None), None
