# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the
# `#[host(class)]` declarations of `lk-agents-abi`. `_core.pyi` re-exports it.

from __future__ import annotations

from collections.abc import Mapping, Sequence

from datetime import date
from livekit.agents.beta.workflows import _TransferHost
from livekit.agents.voice.avatar import AudioReceiver, VideoGenerator, _AvatarMedia
from livekit.agents.voice.room_io import RemoteParticipant, Room, TrackPublication

from .._core import PlaybackInfo, PlaybackTracker, RoomEventBus, Sound, _core_RoomSession
from ..types import NotGivenOr
from .amd import AmdConstants
from .avatar import AudioReceiverFrame
from .io_events import AvatarPlaybackSegment, PlaybackSegment
from .session_wire import StartPlan
from .workflows import InstructionsArg

def amd_constants() -> AmdConstants:
    """These are the reference's, not tuning knobs: the default models are what the benchmark
    ran on.
    """

def amd_resolve_max_endpointing_delay(provided: float | None, activity: float | None) -> float:
    """`_resolve_classifier`'s `max_endpointing_delay`: explicit option, else the activity's,
    else the default. `provided` is `None` only when the key was absent: an explicit default
    still beats a disagreeing activity.
    """

class DataStreamAudioOutput:
    """The agent's native audio sink, sending one byte stream per avatar speech segment."""
    def __init__(
        self,
        room: object,
        *,
        destination_identity: str,
        sample_rate: int | None = None,
        wait_remote_track: int | None = None,
        clear_buffer_timeout: float | None = 2.0,
        wait_playback_start: bool = False,
    ) -> None: ...
    @property
    def label(self) -> str: ...
    @property
    def sample_rate(self) -> int | None: ...
    async def aclose(self) -> None:
        """Stop admitting audio, settle startup and pending writes, close streams and retire RPC routes."""

class DataStreamAudioReceiver:
    """The worker's receiver keeps stream identity with each delivered frame and playback report."""
    def __init__(
        self,
        room: Room,
        *,
        sender_identity: str | None = None,
        frame_size_ms: int = 100,
        rpc_max_retries: int = 3,
        on_clear_buffer: AudioReceiver | None = None,
    ) -> None: ...
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
    def __init__(
        self,
        room: _core_RoomSession,
        *,
        receiver: AudioReceiver,
        generator: VideoGenerator,
        media: _AvatarMedia,
        lazy_publish: bool,
    ) -> None: ...
    async def start(self) -> None: ...
    async def wait_for_complete(self) -> None: ...
    def clear_buffer(self, epoch: int) -> None: ...
    async def aclose(self) -> None: ...

class PlayState:
    """A sound's monotone stop and completion handle."""
    def __init__(self, fade_out: float = 0.0) -> None: ...
    @property
    def done(self) -> bool: ...
    @property
    def stop_requested(self) -> bool: ...
    def stop(self) -> bool: ...
    def mark_playout_done(self) -> bool: ...
    async def wait_for_playout(self) -> None: ...

class BackgroundAudioPlayer:
    """Runtime-owned mixer and published background track."""
    def __init__(
        self,
        sample_rate: int = 48000,
        num_channels: int = 1,
        *,
        blocksize: int = 4800,
        stream_timeout_ms: int = 200,
    ) -> None: ...
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
) -> bool: ...

def _core_playback_finished(
    tracker: PlaybackTracker,
    segment: PlaybackSegment,
    playback_position: float,
    interrupted: bool,
    synchronized_transcript: str | None,
) -> bool: ...

def _core_create_room_event_bus() -> RoomEventBus: ...

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
    def __init__(self) -> None: ...
    def add_pushed_samples(self, samples_per_channel: int, sample_rate: int) -> None: ...
    def add_discarded(self, duration: float) -> None: ...
    def snapshot(self, queued: float) -> PlaybackInfo: ...
    def settle(self) -> PlaybackInfo: ...

def plan_session_start(*, job_default: bool, another_session_is_primary: bool) -> StartPlan:
    """The job's recording and whether this session is its primary (records) and hosts the
    remote session protocol; `another_session_is_primary` is the job's claim already taken.
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

class GetDtmfResult:
    """A collected, space-separated keypad sequence."""
    def __init__(self, *, user_input: str) -> None: ...
    @property
    def user_input(self) -> str: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: GetDtmfResult) -> bool: ...
    @staticmethod
    def from_dtmf_inputs(events: Sequence[str]) -> GetDtmfResult: ...

async def send_dtmf_events(room: _core_RoomSession, events: Sequence[str]) -> str: ...

async def receive_dtmf(
    room: _core_RoomSession,
    num_digits: int,
    timeout: float,
    stop_event: str,
) -> GetDtmfResult: ...

class WarmTransfer:
    """A transfer plan and the runtime ownership of its current dial."""
    def __init__(
        self,
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
        twilio: dict[str, object] | None = None,
    ) -> None: ...
    @property
    def instructions(self) -> str: ...
    def tools(self) -> list[dict[str, object]]: ...
    async def run(self, host: _TransferHost, caller: _core_RoomSession) -> str: ...
    async def call_tool(self, name: str, arguments: object) -> None: ...

class ConsoleIo:
    """`lk agent console`'s socket, opened when the session starts."""
    def __init__(self, addr: str) -> None: ...

async def _core_voice_session_start(session: object) -> None: ...

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

async def wait_for_participant_attribute(
    room: _core_RoomSession,
    identity: str,
    attribute: str,
    value: str,
) -> None: ...

async def wait_for_track_publication(
    room: _core_RoomSession,
    identity: str | None,
    kinds: Sequence[int] | None,
    include_local: bool,
    subscribed: bool,
) -> tuple[str | None, str]: ...
