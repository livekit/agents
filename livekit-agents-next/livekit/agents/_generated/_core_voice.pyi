# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping, Sequence
from datetime import date
from typing import Any, Self

from livekit.agents import VAD
from livekit.agents._generated.expressive import ExpressiveOptions
from livekit.agents.beta.workflows import _TransferHost
from livekit.agents.job import JobContext
from livekit.agents.llm import LLM
from livekit.agents.llm.tool_context import ToolChoice
from livekit.agents.stt import STT
from livekit.agents.tts import TTS
from livekit.agents.voice._session_bindings import SessionHost
from livekit.agents.voice.agent_session import TextTransforms, _SessionNodes
from livekit.agents.voice.avatar import AudioReceiver, AvatarSession, VideoGenerator
from livekit.agents.voice.io import (
    AudioInput,
    AudioOutput,
    TextOutput,
    TurnDetectionMode,
    VideoInput,
)
from livekit.agents.voice.room_io import RemoteParticipant, Room, TrackPublication

from .._core import SessionRecorder, Storage, VoiceAgent, _core_ContextActivation, _core_VoiceStore
from ..types import NotGivenOr
from .agent_nodes import NodeFrame
from .amd import AMDConfig, AmdConstants, AMDPredictionEvent
from .avatar import AudioReceiverFrame, MediaFrame
from .avatar_options import AvatarOptions
from .avatar_queue import QueueEffects
from .background_audio_wire import Sound
from .chat_items import ChatContent, ChatItem
from .connect import APIConnectOptions
from .events import AgentState, UserState
from .io_events import AvatarPlaybackSegment, PlaybackSegment
from .llm import ChatChunk
from .metrics import AvatarMetrics
from .rtc_frames import AudioFrame
from .rtc_wire import PlaybackInfo
from .session_options import (
    InterruptionGateVerdict,
    ResolvedSessionOptions,
    SessionKwargs,
    SessionLegs,
    UpdateKwargs,
)
from .session_owner import SessionJobInfo
from .session_wire import DefaultModelPlan, ExceededTurnReply, RunQuestion, VoiceRunOutput
from .stream_info import StreamInfo
from .stt import SpeechEvent
from .telemetry import RecordingOptions, SessionReport, SpanContext
from .tool_options import ToolSpec
from .tool_report import ToolReport
from .transcript import TranscriptChunk
from .turn import EndpointingOptions, InterruptionMode, InterruptionOptions
from .usage import Usage
from .voice_session import (
    DtmfTaskOptions,
    UpdateAgentArgs,
    VoiceCommandOrigin,
    VoiceSessionArgs,
    VoiceTaskValue,
)
from .workflows import CapturePlan, InstructionsArg, TwilioDial

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
        room: Room,
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
        options: AvatarOptions,
        queue_size_ms: int,
        lazy_publish: bool,
    ) -> Self: ...
    @property
    def av_sync(self) -> AVSynchronizer: ...
    async def start(self) -> None: ...
    async def wait_for_complete(self) -> None: ...
    def clear_buffer(self, epoch: int) -> None: ...
    async def aclose(self) -> None: ...

class _core_AvatarLifecycle:
    """`_core_AvatarLifecycle`: an agent's avatar from `AvatarSession.start()` to its close. Created,
    it waits in the background for the avatar participant to join `room` and publish video (once
    `room` is connected) and reports the join as `AvatarMetrics` (`session_started_time` when the
    room was connected, `avatar_joined_time` when the video was published); a session's assistant
    message added with a `playback_latency` is reported as `AvatarMetrics(playback_latency=..)`
    until the close. Each metric reaches the avatar's `metrics_collected` and the session's
    metrics once it runs. Inside a job, the job's shutdown runs the avatar's `aclose`.
    """
    def __new__(
        cls,
        owner: _core_SessionOwner,
        room: Room,
        *,
        identity: str,
        provider: str,
        avatar: AvatarSession,
        job: JobContext | None = None,
    ) -> Self: ...
    async def wait_for_join(self, timeout: float | None) -> None:
        """Until the avatar joined; `timeout` seconds at most (`None` waits as long as it takes)."""
    async def aclose(self) -> None:
        """Stops the join wait and the metrics, then, inside a job and while `room` is connected,
        removes the avatar participant from the room through the job's RoomService
        (`RemoveParticipant`): a participant already gone (`not_found`) is skipped, any other
        failure logged. Once, however many callers: each waits for the same close.
        """

class _core_QueueAudio:
    def __new__(cls, wait_playback_start: bool) -> Self: ...
    def check_open(self) -> None: ...
    def capture(self, frame: AudioFrame, segment: PlaybackSegment, epoch: int) -> QueueEffects: ...
    def flush(self) -> None: ...
    def clear(self) -> QueueEffects: ...
    def playback_finished(
        self,
        playback_position: float,
        interrupted: bool,
        segment: AvatarPlaybackSegment | None,
    ) -> QueueEffects: ...
    def playback_started(self, segment: AvatarPlaybackSegment | None) -> QueueEffects: ...
    async def next_audio(
        self,
    ) -> tuple[int, bool, AvatarPlaybackSegment | None, AudioReceiverFrame | None]: ...
    def close(self) -> QueueEffects: ...

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
    """`_core.PlaybackTracker`: the tracker a host sink reports to (`io.AudioOutput._tracker`),
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

class LlmNodeChunks:
    """The default `llm_node`'s chunks."""
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> ChatChunk: ...

class TtsNodeFrames:
    """The default `tts_node`'s frames."""
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> NodeFrame: ...

class SttNodeEvents:
    """The default `stt_node`'s speech events."""
    def __aiter__(self) -> Self: ...
    async def __anext__(self) -> SpeechEvent: ...

def llm_node(
    llm: LLM[Any],
    chat_ctx: Sequence[ChatItem],
    chat_ctx_host: object,
    *,
    tools: Sequence[ToolSpec],
    tools_host: object | None = None,
    tool_choice: ToolChoice | None = None,
    conn_options: APIConnectOptions | None = None,
    calls: _core_NodeCalls | None = None,
) -> LlmNodeChunks:
    """A completion of `chat_ctx` (its items, and the host's `llm.ChatContext` a host LLM is
    handed) offering `tools` (their specs, and the host's `list[Tool]`). `conn_options`: the
    session's, else the defaults. `calls`: the running agent node's, when it runs this one.
    """

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
    calls: _core_NodeCalls | None = None,
) -> SttNodeEvents: ...

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

class AVSynchronizer:
    """The native synchronizer view returned by an avatar runner. Both media types use the shared
    RTC sources; no language-specific avatar media manager participates in its lifetime.
    """
    async def push(self, frame: MediaFrame, timestamp: float | None = None) -> None: ...
    async def clear_queue(self) -> None: ...
    async def wait_for_playout(self) -> None: ...
    def reset(self) -> None: ...
    @property
    def actual_fps(self) -> float: ...
    @property
    def last_video_time(self) -> float: ...
    @property
    def last_audio_time(self) -> float: ...
    async def aclose(self) -> None: ...

class PlayoutLedger:
    def __new__(cls) -> Self: ...
    def add_pushed_samples(self, samples_per_channel: int, sample_rate: int) -> None: ...
    def add_discarded(self, duration: float) -> None: ...
    def snapshot(self, queued: float) -> PlaybackInfo: ...
    def settle(self) -> PlaybackInfo: ...

class ByteStreamSender:
    def open(
        self,
        info: StreamInfo,
        sender: str,
        chunks: AsyncIterable[bytes | bytearray | memoryview | list[int]],
    ) -> bool: ...

class TextStreamSender:
    def open(self, info: StreamInfo, sender: str, chunks: AsyncIterable[str]) -> bool: ...

class RpcHandler:
    def call(self, caller_identity: str, payload: str) -> str: ...

class _core_RoomRpcRouter:
    """Private room-adapter service; unrelated rooms never share mutable RPC routes."""
    def __new__(cls) -> Self: ...

class _core_VoiceRun:
    def __new__(cls) -> Self: ...
    def record(self, item: ChatItem) -> None: ...
    def settle(self, status: str, output: object | None, question: RunQuestion | None) -> None: ...
    @property
    def status(self) -> str: ...
    @property
    def output(self) -> VoiceRunOutput | None: ...
    @property
    def question(self) -> str | None: ...
    @property
    def verbatim(self) -> str | None: ...
    @property
    def items(self) -> list[ChatItem]: ...

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
    def update_providers(self, providers: Mapping[str, object | None]) -> dict[str, object | None]:
        """Update base provider references; returns only providers whose effective value changed.
        The runtime retains the host objects; the voice core resolves their resource identities.
        """
    def override_providers(
        self,
        layer: int,
        providers: Mapping[str, object | None],
    ) -> dict[str, object | None]:
        """Attach providers to the same override id used by turn settings, before or after start."""
    def release_providers(self, layer: int) -> dict[str, object | None]:
        """Release provider references and return the providers to restore. Idempotent."""
    def provider(self, slot: str) -> object | None:
        """`session.stt` / `tts` / `vad`: the host provider `slot` runs on now, the topmost held
        override's, else the base's; `None` when the slot has none or it was removed.
        """
    def resolved(self, legs: SessionLegs) -> ResolvedSessionOptions:
        """The resolved configuration over the host-supplied legs, as the dict `session.options` reads
        before `start()`.
        """
    @property
    def turn_detection(self) -> str | None:
        """The turn detection mode the session's own settings name (the constructor's and every
        `update_options` before `start()`); `None` when automatic or a detector object.
        """
    @property
    def interruption_mode(self) -> InterruptionMode | None:
        """The interruption mode `VoiceSession(turn_handling=)` set; `None` when the session picks."""

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
    async def report(self) -> SessionReport:
        """The session's part of its report (history and usage), read from the core now, or as its
        loop closed.
        """
    @property
    def current_agent(self) -> object | None:
        """`session.current_agent`: the host's agent the loop installed last, the task's while a
        `VoiceTask` holds the floor.
        """
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
        origin: VoiceCommandOrigin | None = None,
        opens_with_reply: bool = False,
        dtmf: DtmfTaskOptions | None = None,
    ) -> VoiceTaskValue:
        """`VoiceTask.run()`: the task's agent, sent from inside a tool (`origin` is its
        captured tool identity). The agent holds the floor until one of its tools completes with a
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
        tools: Sequence[ToolSpec] | None,
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
        tool_choice: ToolChoice | None = None,
        tools: Sequence[str] | None = None,
        allow_interruptions: bool | None = None,
        chat_ctx: Sequence[ChatItem] | None = None,
        origin: VoiceCommandOrigin | None = None,
    ) -> SpeechHandle:
        """`generate_reply()`: reply now, with no end of turn behind it. The overrides apply to this
        reply only; `tools` narrows the driving agent's tools to these names (its provider-owned
        tools stay), refused on a realtime model as `chat_ctx` is. Sent from the agent's
        `on_enter`, it is that hook's reply.
        """
    def ivr_start(
        self,
        max_silence: float = 5.0,
        tools: Sequence[ToolSpec] | None = None,
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
    ) -> bool:
        """`AMD._run`: the answering-machine detector over its config, classifier LLM, tools and
        optional STT. It listens once the callee answers in the session's room; the verdict comes
        back as `amd_result`. Returns false if another detector is already attached; the host
        must not publish a rejected detector as the session's owner.
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
        origin: VoiceCommandOrigin | None = None,
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
    def tool_update(
        self,
        invocation_id: str,
        operation: int,
        call_id: str,
        report: ToolReport,
    ) -> None:
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
    async def wait_for_idle_and_hold(self, *, origin: VoiceCommandOrigin | None = None) -> int:
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
        origin: VoiceCommandOrigin | None = None,
        host: object | None = None,
    ) -> None:
        """`VoiceSession._run(user_input=)`, answered on the stream: every run event carries `host`,
        the host's run recorder.
        """
    def say(
        self,
        text: str | AsyncIterable[str],
        *,
        audio: AsyncIterable[AudioFrame] | None = None,
        allow_interruptions: bool | None = None,
        add_to_chat_ctx: bool = True,
        origin: VoiceCommandOrigin | None = None,
    ) -> SpeechHandle:
        """`say()`: speak `text` without the LLM, the words given or a stream of them read to its
        end. With `audio`, those frames play instead of the TTS and `text`, a `str`, is still
        committed. The handle is the promise the session fulfils.
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

def default_session_models(
    vad_supplied: bool,
    detector_supplied: bool,
    realtime: bool,
) -> DefaultModelPlan: ...

def user_turn_exceeded_reply() -> ExceededTurnReply: ...

class _core_SessionState:
    """`_core_SessionState`: one delivered state (the history and the agent's and user's states), as
    the host reads it.
    """
    @property
    def history(self) -> list[object]:
        """`session.history`'s items, in history order: the host's own objects, the same ones each
        later state and every event naming the item hand back.
        """
    @property
    def agent_state(self) -> AgentState:
        """`session.agent_state`: `"initializing"` until the first `agent_state_changed`."""
    @property
    def user_state(self) -> UserState:
        """`session.user_state`: `"listening"` until the first `user_state_changed`."""

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

class _core_SessionBindings:
    def __new__(
        cls,
        room: _core_RoomSession,
        args: VoiceSessionArgs,
        *,
        record_audio: bool = False,
    ) -> Self: ...

class _core_SessionOwner:
    def __new__(cls, options: SessionOptions) -> Self: ...
    async def start(
        self,
        storage: _core_SessionStorage,
        host: SessionHost,
        *,
        job: SessionJobInfo | None = None,
    ) -> None:
        """`job`: the job the session runs in, whose trace its spans join under the span current at
        the call, and which alone may read its report. Startup runs in order: the thread's storage
        opens; the host prepares the room and legs; a session of the job (on its room, or the
        console's) claims the job's primary session and sets the job's recording from its plan;
        the start time is set; the job's session opens its audio recording and the job's shutdown
        is made to close it; the recording is ready; the toolsets set up and the voice loop starts.
        """
    @property
    def session(self) -> VoiceSession | None: ...
    @property
    def span(self) -> SpanContext | None:
        """The session's `agent_session` span, from `start()`: the job's current span while the
        session runs.
        """
    @property
    def storage(self) -> _core_SessionStorage | None: ...
    @property
    def current_agent(self) -> object | None: ...
    @property
    def state(self) -> _core_SessionState:
        """The latest state the host's events delivered; before `start()`, an empty history."""
    @property
    def usage(self) -> Usage:
        """Empty before startup; the final snapshot remains readable after close."""
    @property
    def keyterms(self) -> list[str]:
        """No keyterms are applied before a voice driver exists."""
    def update_agent(
        self,
        agent: object,
        frame: UpdateAgentArgs | None,
        *,
        chat_ctx: Sequence[ChatItem] | None = None,
    ) -> None:
        """`update_agent(agent, chat_ctx=)`. Before `start()`, `agent` drives once the session
        starts, on `chat_ctx` unless `start()` names its own; after, `frame` is the handoff and
        `chat_ctx` the context `agent` works on instead of the conversation carried forward.
        """
    def update_options(
        self,
        *,
        stt: NotGivenOr[object | None] = ...,
        tts: NotGivenOr[object | None] = ...,
        vad: NotGivenOr[object | None] = ...,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = ...,
        endpointing_opts: NotGivenOr[EndpointingOptions] = ...,
        interruption: NotGivenOr[InterruptionOptions] = ...,
        min_consecutive_speech_delay: NotGivenOr[float] = ...,
        tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None] = ...,
        expressive: NotGivenOr[bool | ExpressiveOptions | None] = ...,
        keyterms: NotGivenOr[Sequence[str]] = ...,
    ) -> None:
        """`update_options(..)`: change only the supplied settings. A replaced `stt`, `tts` or `vad`
        is installed in place on a running session, without restarting the call or interrupting
        speech; a setting an `override_options` layer holds takes the new value once the layer is
        released. `interruption` merges over the session's and takes no `mode`, which is set with
        `VoiceSession(turn_handling=)`.
        """
    def override_options(
        self,
        *,
        stt: NotGivenOr[object | None] = ...,
        tts: NotGivenOr[object | None] = ...,
        vad: NotGivenOr[object | None] = ...,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = ...,
        endpointing_opts: NotGivenOr[EndpointingOptions] = ...,
        interruption: NotGivenOr[InterruptionOptions] = ...,
        min_consecutive_speech_delay: NotGivenOr[float] = ...,
        tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None] = ...,
    ) -> int:
        """`override_options(..)` entered: a layer over the session's settings and providers, each
        taking its value from the topmost layer that sets it, named by the id returned, refused as
        `update_options` refuses.
        """
    def release_override(self, layer: int) -> None:
        """The `override_options` block ended, however it ended: its settings are released and the
        providers it changed restored. On a closed session the settings still release and the
        restore is refused (`SessionClosing`); there is nothing left to restore.
        """
    def resolved_options(self, legs: SessionLegs) -> ResolvedSessionOptions: ...
    async def aclose(self, reason: str, drain: bool = False) -> None: ...
    def shutdown(self, reason: str, drain: bool = False) -> None: ...
    async def finish_recording(self) -> None:
        """The recording's final result remains available to the job reporter."""
    @property
    def started_at(self) -> float | None:
        """When the session started (wall clock seconds), once its startup prepared it."""
    @property
    def recording_options(self) -> RecordingOptions:
        """What the session records: its job's plan for it, nothing outside a job or before it starts."""
    @property
    def recording_path(self) -> str | None:
        """The session's audio recording file (`audio.ogg` in the job's session directory), when it
        records audio.
        """
    @property
    def recording_started_at(self) -> float | None:
        """When the audio recording's first frame was written."""
    def allows_report(self, job_id: str) -> bool:
        """Whether the job `job_id` may read the session's report: the job the session first started
        in, or any job for a session started outside one.
        """
    def accepts_event(self, kind: str) -> bool:
        """Whether the session's report keeps an event of `kind`: a known kind, its content only when
        the session's recording is not redacted.
        """
    def admit_event(self, kind: str) -> bool:
        """An event of `kind` the report keeps is counted as retained; whether it was accepted."""

class _core_SessionStorage:
    def __new__(
        cls,
        namespace: Storage | None,
        thread_id: str | None,
        lane: str,
        fallback_lane: str,
        initial_history: Sequence[ChatItem] | None,
    ) -> Self: ...
    async def open(self) -> None:
        """Load the supplied lane, or create the ephemeral context store selected by voice policy."""
    @property
    def namespace(self) -> Storage | None: ...
    @property
    def thread_id(self) -> str | None: ...
    @property
    def scope_id(self) -> str: ...
    @property
    def activation(self) -> _core_ContextActivation: ...
    @property
    def voice_store(self) -> _core_VoiceStore | None: ...
    @property
    def history(self) -> list[ChatItem] | None: ...
    async def aclose(self) -> None: ...

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
    async def wait_for_playout(self, *, origin: VoiceCommandOrigin | None = None) -> None:
        """`wait_for_playout()`, and therefore `__await__`, from `origin` (the captured tool invocation inside a tool). Raises from inside a tool this speech waits on; otherwise never raises: a speech
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
    async def wait_reported(self) -> None:
        """Resolves once the session has handed the host this speech's `speech_done` event, so what
        waits on it runs after that event's callbacks; at once for a speech the session ended
        before creating and for a handle constructed directly, and when the session closes for one
        it never reported. Never raises.
        """

def dtmf_capture_plan(confirmation: bool) -> CapturePlan: ...

def email_capture_plan() -> CapturePlan: ...

def email_read_back(email_address: str) -> str: ...

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
    async def recv(self) -> TranscriptChunk | None: ...
    async def aclose(self) -> None: ...

def rtc_wire_names() -> tuple[list[str], list[str], list[str]]: ...

def video_queue_frames() -> int: ...

WARM_TRANSFER_IDENTITY: str

TWILIO_RINGING_TIMEOUT: float

class SessionClosing(RuntimeError):
    """`wait_for_idle()`/`wait_for_idle_and_hold()` refused: the session is closing."""
