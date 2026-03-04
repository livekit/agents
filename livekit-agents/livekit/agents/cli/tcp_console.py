"""TCP console mode: connects to the Go CLI's TCP server for audio I/O and events."""

from __future__ import annotations

import asyncio
import logging
import queue as stdlib_queue
import time

from livekit import rtc
from livekit.protocol.agent_pb import agent_session as agent_pb

from ..voice import io
from ..voice.events import (
    AgentState,
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
    UserState,
    UserStateChangedEvent,
)
from ..voice.tcp_transport import TcpSessionTransport

logger = logging.getLogger(__name__)

# The Go CLI pipeline runs at 48kHz mono; the agent session runs at 24kHz mono.
WIRE_SAMPLE_RATE = 48000
AGENT_SAMPLE_RATE = 24000

# Mappings from Python string states to proto enums
_AGENT_STATE_MAP: dict[AgentState, int] = {
    "initializing": agent_pb.SESSION_AGENT_STATE_INITIALIZING,
    "idle": agent_pb.SESSION_AGENT_STATE_IDLE,
    "listening": agent_pb.SESSION_AGENT_STATE_LISTENING,
    "thinking": agent_pb.SESSION_AGENT_STATE_THINKING,
    "speaking": agent_pb.SESSION_AGENT_STATE_SPEAKING,
}

_USER_STATE_MAP: dict[UserState, int] = {
    "speaking": agent_pb.SESSION_USER_STATE_SPEAKING,
    "listening": agent_pb.SESSION_USER_STATE_LISTENING,
    "away": agent_pb.SESSION_USER_STATE_AWAY,
}

_METRICS_FIELDS = (
    "transcription_delay",
    "end_of_turn_delay",
    "on_user_turn_completed_delay",
    "llm_node_ttft",
    "tts_node_ttfb",
    "e2e_latency",
)


def _metrics_to_proto(metrics: dict | None) -> agent_pb.MetricsReport:
    """Convert a ChatMessage.metrics dict to a MetricsReport proto."""
    if not metrics:
        return agent_pb.MetricsReport()
    kwargs = {k: metrics[k] for k in _METRICS_FIELDS if k in metrics}
    return agent_pb.MetricsReport(**kwargs)


class TcpAudioInput(io.AudioInput):
    """Audio input that receives frames from the TCP transport."""

    def __init__(self) -> None:
        super().__init__(label="TCP Console")
        self._queue: stdlib_queue.Queue[rtc.AudioFrame] = stdlib_queue.Queue()
        self._resampler = rtc.AudioResampler(
            input_rate=WIRE_SAMPLE_RATE,
            output_rate=AGENT_SAMPLE_RATE,
            num_channels=1,
        )

    def push_frame(self, frame: agent_pb.SessionAudioFrame) -> None:
        """Push a proto AudioFrame from the transport (48kHz) into the queue (24kHz)."""
        audio_frame = rtc.AudioFrame(
            data=frame.data,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )
        resampled = self._resampler.push(audio_frame)
        for rf in resampled:
            self._queue.put_nowait(rf)

    async def __anext__(self) -> rtc.AudioFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._queue.get)


class TcpAudioOutput(io.AudioOutput):
    """Audio output that sends frames to the TCP transport."""

    def __init__(self, transport: TcpSessionTransport) -> None:
        super().__init__(
            label="TCP Console",
            next_in_chain=None,
            sample_rate=AGENT_SAMPLE_RATE,
            capabilities=io.AudioOutputCapabilities(pause=True),
        )
        self._transport = transport
        self._resampler = rtc.AudioResampler(
            input_rate=AGENT_SAMPLE_RATE,
            output_rate=WIRE_SAMPLE_RATE,
            num_channels=1,
        )

        self._pushed_duration: float = 0.0
        self._capture_start: float = 0.0
        self._flush_task: asyncio.Task[None] | None = None
        self._playout_done = asyncio.Event()
        self._interrupted_ev = asyncio.Event()
        self._agent_loop: asyncio.AbstractEventLoop | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self._agent_loop is None:
            self._agent_loop = asyncio.get_running_loop()

        if self._flush_task and not self._flush_task.done():
            logger.error("capture_frame called while previous flush is in progress")
            await self._flush_task

        if not self._pushed_duration:
            self._capture_start = time.monotonic()
            self.on_playback_started(created_at=time.time())

        self._pushed_duration += frame.duration

        resampled = self._resampler.push(frame)
        for rf in resampled:
            audio_frame = agent_pb.SessionAudioFrame(
                data=bytes(rf.data),
                sample_rate=WIRE_SAMPLE_RATE,
                num_channels=rf.num_channels,
                samples_per_channel=rf.samples_per_channel,
            )
            msg = agent_pb.AgentSessionMessage(audio_output=audio_frame)
            self._transport.send_message_threadsafe(msg)

    def flush(self) -> None:
        super().flush()
        msg = agent_pb.AgentSessionMessage(
            audio_playback_flush=agent_pb.SessionAudioPlaybackFlush()
        )
        self._transport.send_message_threadsafe(msg)

        if self._pushed_duration:
            if self._flush_task and not self._flush_task.done():
                logger.error("flush called while previous flush is in progress")
                self._flush_task.cancel()

            self._playout_done.clear()
            self._interrupted_ev.clear()
            self._flush_task = asyncio.create_task(self._wait_for_playout())

    def clear_buffer(self) -> None:
        msg = agent_pb.AgentSessionMessage(
            audio_playback_clear=agent_pb.SessionAudioPlaybackClear()
        )
        self._transport.send_message_threadsafe(msg)

        if self._pushed_duration:
            self._interrupted_ev.set()

    def notify_playout_finished(self) -> None:
        if self._agent_loop is not None:
            self._agent_loop.call_soon_threadsafe(self._playout_done.set)
        else:
            self._playout_done.set()

    async def _wait_for_playout(self) -> None:
        wait_done = asyncio.create_task(self._playout_done.wait())
        wait_interrupt = asyncio.create_task(self._interrupted_ev.wait())
        try:
            await asyncio.wait(
                [wait_done, wait_interrupt],
                return_when=asyncio.FIRST_COMPLETED,
            )
            interrupted = wait_interrupt.done() and not wait_done.done()
        finally:
            wait_done.cancel()
            wait_interrupt.cancel()

        if interrupted:
            played = time.monotonic() - self._capture_start
            played = min(max(0, played), self._pushed_duration)
        else:
            played = self._pushed_duration

        self.on_playback_finished(playback_position=played, interrupted=interrupted)

        self._pushed_duration = 0.0
        self._interrupted_ev.clear()


class TcpConsoleSession:
    """Manages the TCP console session lifecycle."""

    def __init__(self, transport: TcpSessionTransport) -> None:
        self._transport = transport
        self._audio_input = TcpAudioInput()
        self._audio_output = TcpAudioOutput(transport)
        self._recv_task: asyncio.Task | None = None
        self._session: object | None = None

    @property
    def audio_input(self) -> TcpAudioInput:
        return self._audio_input

    @property
    def audio_output(self) -> TcpAudioOutput:
        return self._audio_output

    async def start(self) -> None:
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        await self._transport.close()

    def _send_event(self, event: agent_pb.SessionEvent) -> None:
        msg = agent_pb.AgentSessionMessage(event=event)
        self._transport.send_message_threadsafe(msg)

    def on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        old_pb = _AGENT_STATE_MAP.get(event.old_state, agent_pb.SESSION_AGENT_STATE_IDLE)
        new_pb = _AGENT_STATE_MAP.get(event.new_state, agent_pb.SESSION_AGENT_STATE_IDLE)
        self._send_event(
            agent_pb.SessionEvent(
                agent_state_changed=agent_pb.SessionEvent.AgentStateChanged(
                    old_state=old_pb,
                    new_state=new_pb,
                )
            )
        )

    def on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        old_pb = _USER_STATE_MAP.get(event.old_state, agent_pb.SESSION_USER_STATE_LISTENING)
        new_pb = _USER_STATE_MAP.get(event.new_state, agent_pb.SESSION_USER_STATE_LISTENING)
        self._send_event(
            agent_pb.SessionEvent(
                user_state_changed=agent_pb.SessionEvent.UserStateChanged(
                    old_state=old_pb,
                    new_state=new_pb,
                )
            )
        )

    def on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        self._send_event(
            agent_pb.SessionEvent(
                user_input_transcribed=agent_pb.SessionEvent.UserInputTranscribed(
                    transcript=event.transcript,
                    is_final=event.is_final,
                )
            )
        )

    def on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        from ..llm import ChatMessage, FunctionCall, FunctionCallOutput

        item = event.item
        chat_item = agent_pb.ChatContext.ChatItem()
        if isinstance(item, ChatMessage):
            role_map = {
                "developer": agent_pb.DEVELOPER,
                "system": agent_pb.SYSTEM,
                "user": agent_pb.USER,
                "assistant": agent_pb.ASSISTANT,
            }
            pb_role = role_map.get(item.role, agent_pb.ASSISTANT)
            content = []
            if item.text_content:
                content.append(agent_pb.ChatMessage.ChatContent(text=item.text_content))
            pb_msg = agent_pb.ChatMessage(
                id=item.id,
                role=pb_role,
                content=content,
                metrics=_metrics_to_proto(item.metrics),
            )
            chat_item = agent_pb.ChatContext.ChatItem(message=pb_msg)
        elif isinstance(item, FunctionCall):
            chat_item = agent_pb.ChatContext.ChatItem(
                function_call=agent_pb.FunctionCall(
                    id=item.id or "",
                    call_id=item.call_id or "",
                    name=item.name or "",
                    arguments=item.raw_arguments or "",
                )
            )
        elif isinstance(item, FunctionCallOutput):
            chat_item = agent_pb.ChatContext.ChatItem(
                function_call_output=agent_pb.FunctionCallOutput(
                    call_id=item.call_id or "",
                    output=item.output or "",
                    is_error=item.is_error,
                )
            )

        self._send_event(
            agent_pb.SessionEvent(
                conversation_item_added=agent_pb.SessionEvent.ConversationItemAdded(
                    item=chat_item,
                )
            )
        )

    def on_function_tools_executed(self, event: FunctionToolsExecutedEvent) -> None:
        pb_calls = []
        for fc in event.function_calls:
            pb_calls.append(
                agent_pb.FunctionCall(
                    name=fc.name or "",
                    arguments=fc.raw_arguments or "",
                    call_id=fc.call_id or "",
                )
            )
        pb_outputs = []
        for fco in event.function_call_outputs:
            pb_outputs.append(
                agent_pb.FunctionCallOutput(
                    call_id=fco.call_id or "",
                    output=fco.output or "",
                    is_error=fco.is_error,
                )
            )
        self._send_event(
            agent_pb.SessionEvent(
                function_tools_executed=agent_pb.SessionEvent.FunctionToolsExecuted(
                    function_calls=pb_calls,
                    function_call_outputs=pb_outputs,
                )
            )
        )

    def on_metrics_collected(self, event: MetricsCollectedEvent) -> None:
        pass

    def on_error(self, event: ErrorEvent) -> None:
        self._send_event(
            agent_pb.SessionEvent(
                error=agent_pb.SessionEvent.Error(
                    message=str(event.error) if event.error else "Unknown error",
                )
            )
        )

    def register_on_session(self, session: object) -> None:
        self._session = session
        session.on("agent_state_changed", self.on_agent_state_changed)  # type: ignore[attr-defined]
        session.on("user_state_changed", self.on_user_state_changed)  # type: ignore[attr-defined]
        session.on("conversation_item_added", self.on_conversation_item_added)  # type: ignore[attr-defined]
        session.on("user_input_transcribed", self.on_user_input_transcribed)  # type: ignore[attr-defined]
        session.on("function_tools_executed", self.on_function_tools_executed)  # type: ignore[attr-defined]
        session.on("metrics_collected", self.on_metrics_collected)  # type: ignore[attr-defined]
        session.on("error", self.on_error)  # type: ignore[attr-defined]

    async def _recv_loop(self) -> None:
        while True:
            msg = await self._transport.recv_message()
            if msg is None:
                logger.info("TCP connection closed")
                break

            if msg.HasField("audio_input"):
                self._audio_input.push_frame(msg.audio_input)
            elif msg.HasField("audio_playback_finished"):
                self._audio_output.notify_playout_finished()
            elif msg.HasField("request"):
                await self._handle_request(msg.request)

    async def _handle_request(self, req: agent_pb.SessionRequest) -> None:
        if req.HasField("ping"):
            resp = agent_pb.AgentSessionMessage(
                response=agent_pb.SessionResponse(
                    request_id=req.request_id,
                    pong=agent_pb.SessionResponse.Pong(),
                )
            )
            await self._transport.send_message(resp)
        elif req.HasField("send_message"):
            text = req.send_message.text
            if self._session is not None and text:
                self._session.generate_reply(user_input=text)  # type: ignore[attr-defined]
