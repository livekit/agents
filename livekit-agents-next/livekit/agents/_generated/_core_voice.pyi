# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping, Sequence
from typing import Any, Self

from datetime import date

from livekit import rtc
from livekit.agents import VAD
from livekit.agents.beta.workflows import _TransferHost
from livekit.agents.llm import LLM
from livekit.agents.llm.tool_context import ToolChoice
from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.agents.types import APIConnectOptions
from livekit.agents.voice.agent_session import _SessionNodes
from livekit.agents.voice.avatar import AudioReceiver, VideoGenerator, _AvatarMedia
from livekit.agents.voice.io import AudioInput, AudioOutput, TextOutput, VideoInput
from livekit.agents.voice.room_io import RemoteParticipant, Room, TrackPublication
from livekit.rtc import AudioFrame

from .._core import SessionRecorder, VoiceAgent
from ..types import NotGivenOr, TimedString
from .agent_nodes import NodeFrame
from .amd import AMDConfig, AMDPredictionEvent, AmdConstants
from .avatar import AudioReceiverFrame
from .background_audio_wire import Sound
from .chat_items import ChatContent, ChatItem
from .io_events import AvatarPlaybackSegment, PlaybackSegment
from .metrics import AvatarMetrics
from .rtc_wire import PlaybackInfo
from .session_options import (
    InterruptionGateVerdict,
    ResolvedSessionOptions,
    SessionKwargs,
    SessionLegs,
    UpdateKwargs,
)
from .session_wire import StartPlan
from .stt import SpeechEvent
from .tool_report import ToolReport
from .turn import InterruptionMode
from .usage import Usage
from .voice_session import UpdateAgentArgs, VoiceSessionArgs
from .workflows import InstructionsArg, TwilioDial

def amd_constants() -> AmdConstants:
    """These are the reference's, not tuning knobs: the default models are what the benchmark
    ran on.
    """

def amd_resolve_max_endpointing_delay(provided: float | None, activity: float | None) -> float:
    """`_resolve_classifier`'s `max_endpointing_delay`: explicit option, else the activity's,
    else the default. `provided` is `None` only when the key was absent: an explicit default
    still beats a disagreeing activity.
    """

def amd_prediction_is_human(prediction: AMDPredictionEvent) -> bool:
    """`AMDPredictionEvent.is_human`."""

def amd_prediction_is_machine(prediction: AMDPredictionEvent) -> bool:
    """`AMDPredictionEvent.is_machine`."""

class DataStreamAudioOutput:
    """The agent's native audio sink, sending one byte stream per avatar speech segment."""
    def __new__(
        cls,
        room: object,
        *,
        destination_identity: str,
        sample_rate: int | None = None,
        wait_remote_track: int | None = None,
        clear_buffer_timeout: float | None = 2.0,
        wait_playback_start: bool = False,
    ) -> Self: ...
    @property
    def label(self) -> str: ...
    @property
    def sample_rate(self) -> int | None: ...
    async def aclose(self) -> None:
        """Stop admitting audio, settle startup and pending writes, close streams and retire RPC routes."""

class DataStreamAudioReceiver:
    """The worker's receiver keeps stream identity with each delivered frame and playback report."""
    def __new__(
        cls,
        room: Room,
        *,
        sender_identity: str | None = None,
        frame_size_ms: int = 100,
        rpc_max_retries: int = 3,
        on_clear_buffer: AudioReceiver | None = None,
    ) -> Self: ...
    async def start(self) -> None: ...
    async def next_audio(
        self,
    ) -> tuple[int, bool, AvatarPlaybackSegment | None, AudioReceiverFrame | None]: ...
    def notify_playback_started(self, segment: AvatarPlaybackSegment) -> None: ...
    def notify_playback_finished(
        self,
        segment: AvatarPlaybackSegment,
        playback_position: float,
        interrupted: bool,
    ) -> None: ...
    def aclose(self) -> None: ...

class AvatarRunner:
    """The avatar worker's native queue owner. Its decisions remain in RunnerCore."""
    def __new__(
        cls,
        room: _core_RoomSession,
        *,
        receiver: AudioReceiver,
        generator: VideoGenerator,
        media: _AvatarMedia,
        lazy_publish: bool,
    ) -> Self: ...
    async def start(self) -> None: ...
    async def wait_for_complete(self) -> None: ...
    def clear_buffer(self, epoch: int) -> None: ...
    async def aclose(self) -> None: ...

class PlayState:
    """A sound's monotone stop and completion handle."""
    def __new__(cls, fade_out: float = 0.0) -> Self: ...
    @property
    def done(self) -> bool: ...
    @property
    def stop_requested(self) -> bool: ...
    def stop(self) -> bool: ...
    def mark_playout_done(self) -> bool: ...
    async def wait_for_playout(self) -> None: ...

class BackgroundAudioPlayer:
    """Runtime-owned mixer and published background track."""
    def __new__(
        cls,
        sample_rate: int = 48000,
        num_channels: int = 1,
        *,
        blocksize: int = 4800,
        stream_timeout_ms: int = 200,
    ) -> Self: ...
    async def start(
        self,
        room: _core_RoomSession,
        draw: float,
        *,
        ambient: Sequence[Sound],
        choose: bool,
        source: str | None = None,
    ) -> str: ...
    def play(
        self,
        sounds: Sequence[Sound],
        draw: float,
        *,
        looping: bool = False,
        ambient: bool = False,
        choose: bool = True,
    ) -> PlayState: ...
    def on_agent_state(
        self,
        sounds: Sequence[Sound],
        draw: float,
        thinking: bool,
        *,
        choose: bool = True,
    ) -> None: ...
    @property
    def closed(self) -> bool: ...
    async def aclose(self) -> None: ...

class PlaybackTracker:
    """`_core.PlaybackTracker`: the tracker a host sink reports to (`io.AudioOutput._core_tracker()`),
    shared with the session that reads it.
    """
    def __new__(cls) -> Self: ...
    def note_flush(self) -> None: ...
    def reset(self) -> None: ...
    async def wait_for_playout(self) -> tuple[float, bool, str | None]:
        """`AudioOutput.wait_for_playout()`: the last event as `(playback_position, interrupted,
        synchronized_transcript)`.
        """

def _core_playback_begin(tracker: PlaybackTracker, segment: PlaybackSegment) -> bool: ...

def _core_playback_current_segment(tracker: PlaybackTracker) -> PlaybackSegment | None: ...

def _core_playback_note_capture(tracker: PlaybackTracker) -> PlaybackSegment: ...

def _core_playback_started(
    tracker: PlaybackTracker,
    segment: PlaybackSegment,
    created_at: float,
) -> bool: ...

def _core_playback_progressed(
    tracker: PlaybackTracker,
    segment: PlaybackSegment,
    started_at: float,
    offset: float,
    duration: float,
) -> bool:
    """Invalid durations are refused before updating playback or notifying observers."""

def _core_playback_finished(
    tracker: PlaybackTracker,
    segment: PlaybackSegment,
    playback_position: float,
    interrupted: bool,
    synchronized_transcript: str | None,
) -> bool: ...

class AgentNodes:
    """`_core.AgentNodes(host)`: the session's registered nodes over the host's `_SessionNodes`, taken
    in the binding of the agent they run for.
    """
    def __new__(cls, host: _SessionNodes) -> Self: ...

class _core_NodeCalls:
    ...

class TtsNodeFrames:
    """The default `tts_node`'s frames."""
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> NodeFrame: ...

class SttNodeEvents:
    """The default `stt_node`'s speech events."""
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> SpeechEvent: ...

def tts_node(
    tts: TTS[Any],
    text: AsyncIterable[str],
    *,
    conn_options: APIConnectOptions | None = None,
    calls: _core_NodeCalls | None = None,
) -> TtsNodeFrames:
    """`conn_options`: the node's, else the provider's own policy (models.md), else the
    defaults. `calls`: the running agent node's, when it runs this one.
    """

def stt_node(
    stt: STT[Any],
    audio: AsyncIterable[AudioFrame],
    *,
    conn_options: APIConnectOptions | None = None,
    vad: VAD[Any] | None = None,
) -> SttNodeEvents: ...

class SpeechHandle:
    def __new__(cls, *, speech_id: str = "", allow_interruptions: bool = True) -> Self:
        """A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
        behind it.
        """
    @property
    def host(self) -> object | None:
        """The host's object for this speech, as it set it; `None` until then."""
    @host.setter
    def host(self, host: object) -> None:
        """Set by the host when it creates its handle over this one: from then on every event, tool
        and hook naming the speech hands back this object.
        """
    @property
    def id(self) -> str:
        """`speech_handle.id`. Empty until the loop has minted `speech_{n}`; a detached handle answers
        the id it was constructed with.
        """
    @property
    def num_steps(self) -> int: ...
    @property
    def chat_items(self) -> list[ChatItem]:
        """What this speech committed to the conversation; a copy."""
    @property
    def scheduled(self) -> bool: ...
    @property
    def interrupted(self) -> bool:
        """`True` from the tick `interrupt()` was called until the speech is done, and once the loop
        has interrupted the speech for any reason (an interruption, a supersede, the close). A done
        speech reads the loop's verdict.
        """
    def done(self) -> bool: ...
    @property
    def allow_interruptions(self) -> bool:
        """The core's last accepted `allow_interruptions`."""
    @allow_interruptions.setter
    def allow_interruptions(self, value: bool) -> None:
        """Enqueues the change; the accepted view changes when the loop applies it."""
    def exception(self) -> BaseException | None:
        """The exception the speech failed with: the host's own object when a provider raised one."""
    def interrupt(self, *, force: bool = False) -> None:
        """`interrupt(force=)`, in the reference's order. If the speech does not exist yet, the
        command is sent when the loop creates it.
        """
    def end_call(self) -> None:
        """`EndCallTool`: close the session once this speech is done, or once a realtime model's own
        reply to the tool result is. Sent to this speech's own session.
        """
    def hold_interruptions(self) -> None:
        """`_hold_interruptions`: disallow interruptions until every hold taken is released. Raises
        when the speech is already interrupted, without counting the attempt.
        """
    def release_interruptions(self) -> None:
        """`_release_interruptions`. Idempotent past zero, like the reference's."""
    async def wait_for_playout(self, *, origin: str | None = None) -> None:
        """`wait_for_playout()`, and therefore `__await__`, from `origin` (`"tool:<call_id>"` inside a
        tool). Raises from inside a tool this speech waits on; otherwise never raises: a speech
        the session ended before creating, or a handle constructed directly, has nothing to play.
        """
    async def wait_for_generation(self, step: int | None) -> None:
        """`_wait_for_generation(step_idx)`: one step of the turn, not the whole thing; `None` is the
        latest. No generation, no such step and a never-created speech are distinct errors.
        """
    async def wait_for_interrupted(self) -> None:
        """Resolves when the speech is interrupted, and only then: a speech that finishes normally
        never resolves this, so `wait_if_not_interrupted` can race it.
        """
    async def wait_until_created(self) -> None:
        """Resolves once the loop has created the speech and put it in the queue."""

class RoomEventBus:
    """The fan-out a host room creates and hands its SDK callbacks."""
    def __new__(cls) -> Self: ...

def _core_room_track_subscribed(
    bus: RoomEventBus,
    participant: RemoteParticipant,
    publication: TrackPublication,
) -> None: ...

def _core_room_track_unsubscribed(
    bus: RoomEventBus,
    participant: RemoteParticipant,
    publication: TrackPublication,
) -> None: ...

def _core_room_track_published(
    bus: RoomEventBus,
    participant: RemoteParticipant,
    publication: TrackPublication,
) -> None: ...

def _core_room_track_unpublished(
    bus: RoomEventBus,
    participant: RemoteParticipant,
    publication: TrackPublication,
) -> None: ...

def _core_room_local_track_published(bus: RoomEventBus, publication: TrackPublication) -> None: ...

def _core_room_participant_connected(bus: RoomEventBus, participant: RemoteParticipant) -> None: ...

def _core_room_participant_active(bus: RoomEventBus, participant: RemoteParticipant) -> None: ...

def _core_room_participant_attributes_changed(
    bus: RoomEventBus,
    participant: RemoteParticipant,
) -> None: ...

def _core_room_participant_disconnected(
    bus: RoomEventBus,
    participant: RemoteParticipant,
    reason: str,
) -> None: ...

def _core_room_connection_state_changed(bus: RoomEventBus, state: str) -> None: ...

def _core_room_disconnected(bus: RoomEventBus, reason: str) -> None: ...

def _core_room_sip_dtmf_received(
    bus: RoomEventBus,
    digit: str,
    participant: str | None,
) -> None: ...

class PlayoutLedger:
    def __new__(cls) -> Self: ...
    def add_pushed_samples(self, samples_per_channel: int, sample_rate: int) -> None: ...
    def add_discarded(self, duration: float) -> None: ...
    def snapshot(self, queued: float) -> PlaybackInfo: ...
    def settle(self) -> PlaybackInfo: ...

class ByteStreamSender:
    def open(
        self,
        info: rtc.TextStreamInfo | rtc.ByteStreamInfo,
        sender: str,
        chunks: AsyncIterable[bytes | bytearray | memoryview | list[int]],
    ) -> bool: ...

class TextStreamSender:
    def open(
        self,
        info: rtc.TextStreamInfo | rtc.ByteStreamInfo,
        sender: str,
        chunks: AsyncIterable[str],
    ) -> bool: ...

class RpcHandler:
    def call(self, caller_identity: str, payload: str) -> str: ...

class SessionOptions:
    """`AgentSession(**kwargs)`'s options before the session holds them: the raw kwargs, read once,
    and the session layer they became (plus every `update_options` so far).
    """
    def __new__(cls, kwargs: SessionKwargs) -> Self: ...
    def update(self, update: UpdateKwargs) -> None:
        """`update_options(..)` before the session runs, on the same `merge` the driver uses."""
    def override_options(self, update: UpdateKwargs) -> int:
        """`override_options(..)` before the session runs: a layer over the base, named by the id
        returned. The session started over these options holds it until its release.
        """
    def release_override(self, layer: int) -> None:
        """The `override_options` block ended before the session ran."""
    def resolved(self, legs: SessionLegs) -> ResolvedSessionOptions:
        """The resolved configuration over the host-supplied legs, as the dict `session.options` reads
        before `start()`.
        """

class VoiceSession:
    """`voice/agent_session.py`'s `AgentSession`: assembled at construction, started at `start()`,
    closed at `aclose()`; every other method is one command to its loop.
    """
    def __new__(
        cls,
        room: _core_RoomSession,
        args: VoiceSessionArgs,
        options: SessionOptions | None = None,
    ) -> Self:
        """Assemble a session; the loop begins at `start()`. `room` supplies runtime room I/O, or no
        RTC attachment; `options` is the session layer `SessionOptions(kwargs)` built (none is an
        empty session layer).
        """
    async def start(self) -> None:
        """Open the providers and run the loop beside the host event drain. Resolves once the host
        has handled the opening events.
        """
    @property
    def usage(self) -> Usage:
        """`session.usage`: the latest snapshot of the session's usage, settled once it closed."""
    @property
    def audio_input(self) -> RoomAudioInput | None:
        """The room's microphone leg, unless `RoomIoConfig(audio_input=False)`, else `None`. A fresh
        handle each call over the same leg; safe before or after `start()`.
        """
    @property
    def video_input(self) -> RoomVideoInput | None:
        """The room's video leg, when `RoomIoConfig(video_input=True)` asked for one, else `None`."""
    def attach_recorder(self, recorder: SessionRecorder | None) -> None:
        """Record the session into `recorder`; `None` detaches. Safe before or after `start()`; the
        file's `finish` stays the host's.
        """
    def update_models(
        self,
        *,
        llm: NotGivenOr[LLM[Any] | None] = ...,
        tts: NotGivenOr[TTS[Any] | None] = ...,
        stt: NotGivenOr[STT[Any] | None] = ...,
        vad: NotGivenOr[VAD[Any] | None] = ...,
        llm_node: AgentNodes | None = None,
        tts_node: AgentNodes | None = None,
        stt_node: AgentNodes | None = None,
    ) -> None:
        """`Agent.update_options(stt=.., vad=.., llm=.., tts=..)` on a running agent: an in-place
        provider swap, not a handoff. Each leg is three-state; a given one goes under the agent's
        node when it overrides one.
        """
    async def update_agent(self, frame: UpdateAgentArgs) -> bool:
        """The handoff: replace the agent, keep the session. Resolves `True` once the agent is
        installed (`on_enter` issued, not returned), `False` when it was refused (closing).
        """
    async def run_voice_task(
        self,
        frame: UpdateAgentArgs,
        *,
        origin: str | None = None,
        opens_with_reply: bool = False,
    ) -> object:
        """`VoiceTask.run()`: the task's agent, sent from inside a tool (`origin` is its
        `"tool:<call_id>"`). The agent holds the floor until one of its tools completes with a
        value the host validated, which this resolves to; the parent comes back without its hooks.
        A cancelled await ends the task, and every task it started, and puts the parent back.
        """
    async def collect_dtmf(
        self,
        num_digits: int,
        ask_for_confirmation: bool,
        timeout: float,
        stop_event: str,
    ) -> str:
        """`GetDtmfTask`'s keypad half: the keys the linked participant presses, collected until
        `num_digits` arrived and a `timeout` (seconds) passed without a key, or `stop_event` was
        pressed. Resolves to the digits, space-separated; raises `VoiceTaskFailed` when the
        timeout fired short, and is cancelled when the session closes or another collection
        starts. With `ask_for_confirmation` the digits are read back instead and the task's
        `confirm_inputs` tool completes it. A cancelled await stops the collection.
        """
    def report_avatar_metrics(self, metrics: AvatarMetrics) -> None:
        """An `AvatarSession`'s own measurements (its join and playback latency), recorded with the
        session's metrics.
        """
    def set_audio_input(self, source: AudioInput | None) -> None:
        """`session.input.audio = ..` after `start()`; `None` detaches."""
    def set_video_input(self, source: VideoInput | None) -> None:
        """`session.input.video = ..` after `start()`: the video pump reads the new source, or stops
        on `None`.
        """
    def set_audio_output(self, sink: AudioOutput | None) -> None:
        """Select the destination for subsequent output segments. Current work retains its admitted
        binding, including when the same endpoint object is selected again.
        """
    def set_transcription_output(self, sink: TextOutput | None) -> None:
        """`session.output.transcription = ..` after `start()`: the next reply forwards into the new
        leg; one in flight keeps the leg it started with.
        """
    def update_tools(
        self,
        tools: Sequence[dict[str, object]] | None,
        *,
        tools_host: object | None,
        voice_agent: VoiceAgent | None,
    ) -> None:
        """`Agent.update_tools()` on a running agent. All three parts go together: the agent hosts
        its tools.
        """
    def update_turn_options(self, update: UpdateKwargs) -> None:
        """`AgentSession.update_options(..)` on a running session: merged into its session layer and
        re-resolved; `expressive` reaches the next reply, `keyterms` replace the user's list. A turn
        detector object is refused: swapping the detector of a running session is not ported.
        """
    def override_options(self, update: UpdateKwargs) -> int:
        """`override_options(..)` on a running session: a layer over the base until
        `release_override` names the id returned. Refused as `update_turn_options` refuses.
        """
    def release_override(self, layer: int) -> None:
        """The `override_options` block ended, however it ended. A closed session has nothing left
        to restore.
        """
    def resolved_options(self) -> ResolvedSessionOptions | None:
        """The core's accepted configuration; `None` for a session built without options."""
    def push_video(self, content: ChatContent) -> bool:
        """`AgentActivity.push_video(frame)`: one sampled frame, on its way to the model. `False`
        when the loop did not take it (not started or closed); a non-image part raises.
        """
    def generate_reply(
        self,
        *,
        user_input: str | None = None,
        instructions: str | None = None,
        allow_interruptions: bool | None = None,
        tool_choice: ToolChoice | None = None,
        chat_ctx: Sequence[ChatItem] | None = None,
        tools: Sequence[dict[str, object]] | None = None,
        tools_host: object | None = None,
        voice_agent: VoiceAgent | None = None,
        on_enter: bool = False,
        origin: str | None = None,
    ) -> SpeechHandle:
        """`AgentSession.generate_reply()`: reply now, with no end of turn behind it. The overrides
        apply to this reply only; `tools=`, `tools_host=` and `voice_agent=` go together.
        """
    def ivr_start(
        self,
        max_silence: float = 5.0,
        tools: Sequence[dict[str, object]] | None = None,
        *,
        tools_host: object | None = None,
    ) -> None:
        """`IVRActivity.start()`: `max_silence` is `max_silence_duration` in seconds; `tools` is
        `IVRActivity.tools`. A second start is journalled, not run.
        """
    def attach_amd(
        self,
        cfg: AMDConfig,
        llm: LLM[Any],
        tools: object | None,
        stt: STT[Any] | None,
        participant_identity: str | None,
    ) -> None:
        """`AMD._run`: the answering-machine detector over its config, classifier LLM, tools and
        optional STT. It listens once the callee answers in the session's room; the verdict comes
        back as `amd_result`. A second attach is refused.
        """
    async def aclose(self, *, reason: str = "user_initiated", drain: bool = False) -> None:
        """`_aclose_impl`: the hooks' `closing`, the close itself, the turn-detector stream, the
        remote session host, the room io, then the session's toolsets. Idempotent: a second call
        waits for the first.
        """
    def shutdown(self, *, drain: bool = True) -> None:
        """`AgentSession.shutdown(drain=)`: `aclose` without waiting for it."""
    async def set_participant(self, identity: str | None) -> None:
        """`RoomIO.set_participant(identity)`: point the audio input and the user transcript at
        someone else. `None` unlinks.
        """
    def filler_start(
        self,
        call_id: str,
        speech_id: str,
        *,
        delay: float,
        interval: float | None = None,
        max_steps: int | None = None,
        text: str | None = None,
        source: object | None = None,
    ) -> int:
        """`RunContext.with_filler(..)` entered for tool `call_id` of `speech_id`: after `delay` s
        idle say `text` (or ask `source`), then repeat every `interval`. Returns the scope's number.
        """
    def filler_stop(self, id: int) -> None:
        """`with_filler`'s exit."""
    def clear_user_turn(self) -> None:
        """`AgentSession.clear_user_turn()`: drop the open turn and everything accumulated for it."""
    def update_instructions(self, instructions: str) -> None:
        """`Agent.update_instructions()` on the agent that is already driving."""
    def set_io_enabled(self, leg: str, enabled: bool) -> None:
        """`input.set_audio_enabled` / `output.set_audio_enabled` /
        `output.set_transcription_enabled`: `leg` is `"audio_input"`, `"audio_output"` or
        `"transcription"`.
        """
    def tool_update(self, call_id: str, report: ToolReport) -> None:
        """`RunContext.update()` after the first (an `update` outcome), or a detached tool's end."""
    def update_chat_ctx(self, items: Sequence[ChatItem]) -> None:
        """`Agent.update_chat_ctx()` on a running agent: replace the context."""
    async def interrupt(self, *, force: bool = False) -> None:
        """`AgentSession.interrupt(force=)`: the current and queued speeches up to the first
        uninterruptible; answers when they are done, or with the refusal.
        """
    async def drain(self) -> str | None:
        """`AgentSession.drain()`: the driving agent's `on_exit`, then scheduling paused until no
        speech is left. Answers `None`, `"activity_changing"` or `"closing"`.
        """
    async def wait_for_idle(self, *, hold: int | None = None) -> None:
        """`AgentSession.wait_for_idle()`: answers when the session is idle (no hold but `hold`)."""
    async def wait_for_idle_and_hold(self, *, origin: str | None = None) -> int:
        """`RunContext.foreground()`: wait for idle, then hold the floor until `release_idle_hold`.
        Answers the hold's id.
        """
    def release_idle_hold(self, hold: int) -> None:
        """`_wait_for_idle_and_hold`'s exit. A hold the loop no longer has is a no-op."""
    def claim_user_turn(self) -> None:
        """`AgentSession._claim_user_turn()`'s enter: pins the user state to `speaking` and holds
        `wait_for_idle` open. Reentrant.
        """
    def release_user_turn(self) -> None:
        """`_claim_user_turn()`'s exit: the last release re-derives the user state."""
    def run(
        self,
        *,
        user_input: str | None = None,
        origin: str | None = None,
        host: object | None = None,
    ) -> None:
        """`VoiceSession._run(user_input=)`, answered on the stream: every run event carries `host`,
        the host's run recorder.
        """
    def say(
        self,
        text: str,
        *,
        allow_interruptions: bool | None = None,
        add_to_chat_ctx: bool = True,
        origin: str | None = None,
    ) -> SpeechHandle:
        """`AgentSession.say(text)`: speak these words, no LLM. The handle is the promise the loop
        fulfils.
        """
    def say_stream(
        self,
        text: AsyncIterable[str],
        *,
        allow_interruptions: bool | None = None,
        add_to_chat_ctx: bool = True,
        origin: str | None = None,
    ) -> SpeechHandle:
        """`AgentSession.say(AsyncIterable[str])`: the speech reads `text` to its end."""
    def say_audio(
        self,
        text: str,
        audio: AsyncIterable[AudioFrame],
        *,
        allow_interruptions: bool | None = None,
        add_to_chat_ctx: bool = True,
        origin: str | None = None,
    ) -> SpeechHandle:
        """`AgentSession.say(text, audio=..)`: pre-synthesised frames instead of TTS; `text` is still
        committed.
        """
    def ivr_final(self, text: str) -> None:
        """`_start_ivr_detection(transcript)`: a final the navigator counts as if heard."""
    def amd_listening(self) -> bool:
        """`AMD.started`: the detector listens (the callee answered, no verdict out yet)."""
    def amd_execute(self) -> None:
        """`AMD.execute()`'s interrupt and `_resume_authorization()`, after its IVR hand-off."""
    def amd_close(self) -> None:
        """`AMD.aclose()`. Idempotent."""
    def keyterms(self) -> list[str] | None:
        """`session.keyterms`: the list the recogniser was last handed; `None` before any push."""
    def agent_chat_ctx(self) -> list[ChatItem]:
        """`Agent.chat_ctx` of the driving agent; empty before `start()`."""
    async def commit_user_turn(
        self,
        *,
        transcript_timeout: float = 2.0,
        stt_flush_duration: float = 2.0,
        skip_reply: bool = False,
    ) -> str:
        """`AgentSession.commit_user_turn(..)`: flush, wait for the final, commit, answer with the
        transcript. Runs whether or not awaited; a newer commit supersedes it.
        """

def plan_session_start(*, job_default: bool, another_session_is_primary: bool) -> StartPlan:
    """The job's recording and whether this session is its primary (records) and hosts the
    remote session protocol; `another_session_is_primary` is the job's claim already taken.
    """

def rt_server_turn_detection(
    *,
    caps_turn_detection: bool,
    can_disable_turn_detection: bool,
    vad: bool,
    turn_detection: str | None,
    streaming_detector: bool,
    interruption_given: bool,
) -> bool:
    """`AgentActivity._rt_turn_detection_enabled` from the host's reads. `turn_detection` is the
    explicit mode string (agent over session), `None` when unsaid or a detector object;
    `streaming_detector` says it was a streaming detector.
    """

def resolve_interruption_detector(
    *,
    stt_aligned_transcript: bool | None,
    stt_streaming: bool | None,
    realtime: bool,
    rt_server_turn_detection: bool,
    has_vad: bool,
    turn_detection: str | None,
    allow_interruptions: bool,
    agent_mode: InterruptionMode | None,
    session_mode: InterruptionMode | None,
    hosted: bool,
    dev_mode: bool,
) -> InterruptionGateVerdict:
    """`AgentActivity._resolve_interruption_detection`: whether to build the detector, and if not,
    why. Constructing it stays the host's.
    """

def validate_card_number(card_number: str) -> bool:
    """The Luhn checksum over an all-digit string."""

def card_expired(month: int, year: int, today: date) -> bool:
    """Whether a two-digit expiry year and its month precede `today`'s month."""

def normalize_date_of_birth(year: int, month: int, day: int, today: date) -> date:
    """A two-digit year is this century's when it is not past this year's two digits, else the
    last century's; an invalid or future date is refused.
    """

def normalize_email_address(email: str) -> str:
    """Stripped, and refused unless it reads `local@label.….tld`."""

def normalize_phone_number(phone_number: str) -> str:
    """Whitespace, hyphens, parentheses and dots dropped, and refused unless it reads `+?[1-9]`
    followed by six to fourteen digits.
    """

def dtmf_code(symbol: str) -> int:
    """Digits are themselves, `*` is 10, `#` is 11 and `A`-`D` are 12-15."""

def format_dtmf(symbols: Sequence[str]) -> str:
    """The space-joined digits that fill the DTMF read-back's `{digits}` hole."""

async def send_dtmf_events(room: _core_RoomSession, events: Sequence[str]) -> str: ...

async def receive_dtmf(
    room: _core_RoomSession,
    num_digits: int,
    timeout: float,
    stop_event: str,
) -> str: ...

class WarmTransfer:
    """A transfer plan and the runtime ownership of its current dial."""
    def __new__(
        cls,
        *,
        sip_call_to: str | None = None,
        sip_trunk_id: NotGivenOr[str | None] = ...,
        has_sip_connection: bool = False,
        sip_number: NotGivenOr[str] = ...,
        sip_headers: Mapping[str, str] | None = None,
        dtmf: str | None = None,
        ringing_timeout: float | None = None,
        instructions: NotGivenOr[InstructionsArg] = ...,
        conversation: Sequence[tuple[str, str]] | None = None,
        env_sip_trunk: str | None = None,
        env_sip_number: str = "",
        twilio: TwilioDial | None = None,
    ) -> Self: ...
    @property
    def instructions(self) -> str: ...
    def tools(self) -> list[dict[str, object]]: ...
    async def run(self, host: _TransferHost, caller: _core_RoomSession) -> str: ...
    async def call_tool(self, name: str, arguments: object) -> None: ...

class ConsoleIo:
    """`lk agent console`'s socket, opened when the session starts."""
    def __new__(cls, addr: str) -> Self: ...

class RoomAudioInput:
    """`_core.RoomAudioInput`: the room's microphone leg, taken as itself wherever the host passes
    an audio input.
    """
    @property
    def reading(self) -> str | None: ...
    async def next_frame(self) -> AudioFrame | None: ...
    async def aclose(self) -> None: ...

class RoomVideoInput:
    """`_core.RoomVideoInput`: the room's video leg, taken as itself wherever the host passes a video
    input.
    """
    @property
    def reading(self) -> str | None: ...
    async def next_frame(self) -> object | None: ...
    async def aclose(self) -> None: ...

class _core_RoomSession:
    """`_core_RoomSession`: what a session's room I/O is built from, handed to the host and taken
    back as itself.
    """

def _core_create_room_session(room: Room | None) -> _core_RoomSession: ...

async def wait_avatar_join(room: _core_RoomSession, identity: str) -> tuple[float, float]:
    """Wait for the supplied room's connection, avatar participant and subscribed video track."""

async def wait_for_participant(
    room: _core_RoomSession,
    identity: str | None,
    kinds: Sequence[int] | None,
    include_local: bool,
) -> str | None: ...

async def wait_for_agent(room: _core_RoomSession, name: str | None) -> str: ...

async def wait_for_track_publication(
    room: _core_RoomSession,
    identity: str | None,
    kinds: Sequence[int] | None,
    include_local: bool,
    subscribed: bool,
) -> tuple[str | None, str]: ...

class NodeText:
    async def recv(self) -> str | None: ...
    async def aclose(self) -> None: ...

class NodeAudio:
    async def recv(self) -> AudioFrame | None: ...
    async def aclose(self) -> None: ...

class NodeTranscript:
    async def recv(self) -> TimedString | None: ...
    async def aclose(self) -> None: ...
