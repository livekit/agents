"""The remote session protocol a client drives a call through: ``lk agent simulate`` and the
dashboard over the room's ``lk.agent.session`` byte streams, ``lk agent console`` over its socket.
``_core.RemoteSessionHost`` answers the requests; this module is its transport on a room, its
view of the session and the session events it forwards."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any, ClassVar

from google.protobuf.json_format import MessageToDict

from livekit import rtc

from .. import _core, llm, utils
from .._core_view import to_core_items
from ..job import get_job_context
from ..llm.chat_context import Instructions
from ..log import logger
from ..version import __version__

if TYPE_CHECKING:
    from .agent_session import LiveSession
    from .events import FunctionToolsExecutedEvent
    from .run_result import RunResult

TOPIC = _core.session_messages_topic()


class SessionHost:
    """Serves ``session`` to a remote client over ``transport``: forwards the session's events
    and answers the client's requests (``_core.RemoteSessionHost``'s port)."""

    # the core's session events the client is sent, and the conversation items among them
    EVENTS: ClassVar[frozenset[str]] = frozenset(
        {
            "agent_state_changed",
            "user_state_changed",
            "user_input_transcribed",
            "conversation_item_added",
            "tool_execution_updated",
            "agent_false_interruption",
            "error",
        }
    )
    ITEMS: ClassVar[frozenset[str]] = frozenset(
        {"message", "function_call", "function_call_output", "agent_handoff", "agent_config_update"}
    )
    DRAIN: ClassVar[float] = 3.0

    def __init__(self, session: LiveSession[Any], transport: Any, console: Any = None) -> None:
        self._session = session
        self._transport = transport
        self._console = console
        self._host = _core.RemoteSessionHost()
        self._host.set_port(self)
        # answered through a hook, so not among the core's events
        session.on("function_tools_executed", self._tools_executed)

    @classmethod
    def on_room(cls, session: LiveSession[Any], room: rtc.Room) -> SessionHost:
        return cls(session, _RoomTransport(room))

    async def start(self) -> None:
        await self._host.start(self._transport, self._console)

    async def aclose(self) -> None:
        if cancelled := await self._host.aclose(self.DRAIN):
            logger.warning(
                "cancelled in-flight session requests while closing",
                extra={"request_types": cancelled},
            )

    def forward(self, wire: dict[str, Any]) -> None:
        """Send one of the core's session events to the client."""
        kind = wire["type"]
        if kind not in self.EVENTS:
            return
        if kind == "conversation_item_added":
            if wire["item"].type not in self.ITEMS:
                return
        elif kind == "error":
            wire = {"message": str(wire["error"]) or "Unknown error"}
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
            logger.exception("failed to forward a session event to the remote client")

    # --- what the client asks of the session (`RemoteSessionHost`'s port) -----------------

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
        from livekit.protocol import agent_simulation as sim_pb

        from ..simulation import SimulationVerdict

        job_ctx = get_job_context(required=False)
        sim_ctx = job_ctx.simulation_context() if job_ctx is not None else None
        if job_ctx is None or sim_ctx is None:
            return None, None
        try:
            success, reason = provisional
            sim_ctx._begin_finalize(
                simulator_verdict=SimulationVerdict(success=success, reason=reason),
                run=sim_pb.SimulationRun(id=sim_ctx._dispatch.simulation_run_id),
            )
            if (fnc := job_ctx._simulation_end_fnc) is not None:
                if asyncio.iscoroutine(out := fnc(sim_ctx)):
                    await out
        except Exception as e:
            logger.exception("error while executing the on_simulation_end callback")
            return None, str(e)
        verdict = sim_ctx.user_verdict
        return ((verdict.success, verdict.reason) if verdict else None), None


class _RoomTransport:
    """The room's ``lk.agent.session`` byte streams, one message per stream, read in order."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._incoming: utils.aio.Chan[rtc.ByteStreamReader] = utils.aio.Chan()
        self._send_lock = asyncio.Lock()
        self._registered = False

    async def _core_start(self) -> None:
        if not self._registered:
            self._room.register_byte_stream_handler(
                TOPIC, lambda reader, _: self._incoming.send_nowait(reader)
            )
            self._registered = True

    async def _core_send_message(self, data: bytes) -> None:
        if not self._room.isconnected():
            raise RuntimeError("room session transport is closed")
        async with self._send_lock:
            writer = await self._room.local_participant.stream_bytes(
                name=utils.shortuuid("AS_"), topic=TOPIC
            )
            await writer.write(data)
            await writer.aclose()

    def _core_send_message_nowait(self, data: bytes) -> None:
        task = asyncio.ensure_future(self._core_send_message(data))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())

    async def _core_recv(self) -> bytes | None:
        while True:
            try:
                reader = await self._incoming.recv()
            except utils.aio.ChanClosed:
                return None
            try:
                return b"".join([chunk async for chunk in reader])
            except Exception:
                logger.warning("failed to read a session message", exc_info=True)

    async def _core_close(self) -> None:
        self._incoming.close()
        if self._registered:
            with contextlib.suppress(ValueError, AttributeError):
                self._room.unregister_byte_stream_handler(TOPIC)
            self._registered = False
