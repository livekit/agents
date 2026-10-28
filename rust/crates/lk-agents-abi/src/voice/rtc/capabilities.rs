//! SDK room capabilities, declared once. Runtime owns every live queue and operation.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::VideoFrameRef;
use super::events::RoomEventBus;
use super::ports::{ByteStreamSender, RpcHandler, TextStreamSender};
use super::{
    AudioStreamOptions, AudioTrackFeature, ConnectionState, ParticipantIdentity, ParticipantKind,
    PlaybackInfo, RtcError, StreamBytesOptions, StreamTextOptions, TrackPublishOptions, TrackSid,
    TrackSource, Transcription,
};
use crate::{AudioFrame, StreamSource};

/// A transfer's second room; the live session receives an already opened room.
pub trait RtcHost: Send + Sync {
    fn connect<'a>(
        &'a self,
        url: &'a str,
        token: &'a str,
    ) -> crate::BoxFuture<'a, Result<Arc<dyn RtcRoom>, RtcError>>;
}

/// A connected room.
///
/// `connection_state` rather than `isconnected`: the reference's boolean cannot distinguish
/// disconnected from reconnecting, and [`ConnectionState::is_connected`] recovers it exactly.
#[crate::host(proxy(host = "livekit.agents.voice.room_io.Room"))]
pub trait RtcRoom: Send + Sync {
    /// Owned and asked every time: a room's name is `""` until it connects.
    #[host(call, name = "_core_name")]
    fn name(&self) -> Result<String, RtcError>;
    #[host(call, name = "_core_connection_state")]
    fn connection_state(&self) -> Result<ConnectionState, RtcError>;
    /// Lent, so read once: the agent's participant object outlives its connection (the host's
    /// resolves the SDK's on each call).
    #[host(read_once, name = "_core_local")]
    fn local(&self) -> &Arc<dyn LocalParticipant>;

    /// One remote participant by identity, or `None` if they are not in the room.
    #[host(call, name = "_core_participant")]
    fn participant(
        &self,
        id: &ParticipantIdentity,
    ) -> Result<Option<Arc<dyn RemoteParticipant>>, RtcError>;

    /// Every remote participant.
    ///
    /// `RoomIO._init_task` replays
    /// `room.remote_participants.values()` through its `participant_connected` handler to catch
    /// anyone who joined before the handler was registered, and `set_participant` scans the same
    /// map. Without it, linking to a participant who was already in the room would depend on an
    /// event that has already fired.
    #[host(call, name = "_core_remote_participants")]
    fn remote_participants(&self) -> Result<Vec<Arc<dyn RemoteParticipant>>, RtcError>;

    /// Where the room emits its events. `dyn RtcRoom::events` subscribes to it; a host
    /// room pushes into it from its own handlers.
    #[host(read_once, name = "_core_event_bus")]
    fn event_bus(&self) -> &RoomEventBus;

    /// Route every stream opened on `topic` into `streams`. Errors if the topic already has a
    /// handler -- the SDK's `ValueError`, which both callers treat as a warn-and-continue.
    /// `dyn RtcRoom::byte_streams` is the reader's form.
    #[host(call, name = "_core_register_byte_stream_handler")]
    fn register_byte_stream_handler(
        &self,
        topic: &str,
        streams: ByteStreamSender,
    ) -> Result<(), RtcError>;
    /// Idempotent: unregistering a topic with no handler is a no-op, because `aclose` runs on
    /// paths where `register` may never have succeeded.
    #[host(call, name = "_core_unregister_byte_stream_handler")]
    fn unregister_byte_stream_handler(&self, topic: &str) -> Result<(), RtcError>;
    #[host(call, name = "_core_register_text_stream_handler")]
    fn register_text_stream_handler(
        &self,
        topic: &str,
        streams: TextStreamSender,
    ) -> Result<(), RtcError>;
    #[host(call, name = "_core_unregister_text_stream_handler")]
    fn unregister_text_stream_handler(&self, topic: &str) -> Result<(), RtcError>;

    /// The supplied performer owns stream admission; bare SDK proxies do not allocate queues.
    fn byte_streams(
        &self,
        _topic: &str,
    ) -> Result<super::ports::Receiver<super::ports::ByteStream>, RtcError> {
        Err(RtcError::Unsupported("byte_streams"))
    }
    fn text_streams(
        &self,
        _topic: &str,
    ) -> Result<super::ports::Receiver<super::ports::TextStream>, RtcError> {
        Err(RtcError::Unsupported("text_streams"))
    }
    fn wait_for_participant<'a>(
        &'a self,
        _identity: Option<String>,
        _kinds: Option<Vec<i32>>,
        _include_local: bool,
    ) -> crate::BoxFuture<'a, Result<Option<ParticipantIdentity>, RtcError>> {
        Box::pin(async { Err(RtcError::Unsupported("wait_for_participant")) })
    }
    fn wait_for_agent<'a>(
        &'a self,
        _name: Option<String>,
    ) -> crate::BoxFuture<'a, Result<ParticipantIdentity, RtcError>> {
        Box::pin(async { Err(RtcError::Unsupported("wait_for_agent")) })
    }
    fn wait_for_participant_attribute<'a>(
        &'a self,
        _identity: String,
        _attribute: String,
        _value: String,
    ) -> crate::BoxFuture<'a, Result<(), RtcError>> {
        Box::pin(async { Err(RtcError::Unsupported("wait_for_participant_attribute")) })
    }
    fn wait_for_track_publication<'a>(
        &'a self,
        _identity: Option<String>,
        _kinds: Option<Vec<i32>>,
        _include_local: bool,
        _subscribed: bool,
    ) -> crate::BoxFuture<'a, Result<(Option<ParticipantIdentity>, TrackSid), RtcError>> {
        Box::pin(async { Err(RtcError::Unsupported("wait_for_track_publication")) })
    }

    /// Native subscriptions use the runtime fan-out already taken with this room.
    fn events(&self) -> super::ports::Receiver<super::events::RoomEvent> {
        self.event_bus().subscribe()
    }
}

/// The agent itself in the room.
#[crate::host(proxy(host = "livekit.agents.voice.room_io.LocalParticipant"))]
pub trait LocalParticipant: Send + Sync {
    /// Needed by three readers, all comparing against it: `_init_task` sets the agent transcript's
    /// participant to it, `_on_participant_connected` skips participants publishing *on behalf of*
    /// it, and `_on_local_track_published` gates on it. Owned and asked every time: before the room
    /// connects it is `""`, which no reader can match.
    #[host(call, name = "_core_identity")]
    fn identity(&self) -> Result<ParticipantIdentity, RtcError>;

    /// The agent's own kind (`agent`), which `wait_for_participant(include_local=True)` filters.
    #[host(call, name = "_core_kind")]
    fn kind(&self) -> Result<ParticipantKind, RtcError>;

    /// `find_micro_track_id` scans these when the agent is its own transcription target.
    #[host(call, name = "_core_track_publications")]
    fn track_publications(&self) -> Result<Vec<Arc<dyn TrackPublication>>, RtcError>;

    /// Create an `rtc.AudioSource`, wrap it in a `LocalAudioTrack`, publish it.
    ///
    /// Returns both halves because they have different lifetimes and different users: the source
    /// is the data plane (`capture`/`clear_queue`), the publication is identity (`sid`, for the
    /// agent's own transcription) and subscription state.
    #[host(await, name = "_core_publish_audio_track")]
    fn publish_audio_track<'a>(
        &'a self,
        opts: TrackPublishOptions,
    ) -> crate::BoxFuture<'a, Result<(Arc<dyn AudioSource>, Arc<dyn TrackPublication>), RtcError>>;

    #[host(await, name = "_core_unpublish_track")]
    fn unpublish_track<'a>(
        &'a self,
        sid: &'a TrackSid,
    ) -> crate::BoxFuture<'a, Result<(), RtcError>> {
        Box::pin(async move { Err(RtcError::Unsupported("unpublish_track")) })
    }

    /// `ATTRIBUTE_AGENT_STATE` is the only writer; the reference cancels an in-flight call before
    /// issuing the next, so implementations need not serialise.
    #[host(await, name = "_core_set_attributes")]
    fn set_attributes<'a>(
        &'a self,
        attrs: BTreeMap<String, String>,
    ) -> crate::BoxFuture<'a, Result<(), RtcError>>;

    /// The deprecated transcription API. Still published, in parallel with the text stream, by
    /// `_ParticipantLegacyTranscriptionOutput`.
    #[host(await, name = "_core_publish_transcription")]
    fn publish_transcription<'a>(
        &'a self,
        transcription: Transcription,
    ) -> crate::BoxFuture<'a, Result<(), RtcError>>;

    /// Open a text stream and return its **writer**.
    ///
    /// One writer per stream, not one call per text: `_ParticipantStreamTranscriptionOutput`
    /// opens one writer per transcript segment,
    /// writes every delta into it, and closes it with `lk.transcription_final=true` plus the
    /// segment's expression attributes (`_output.py:_flush_task`). A per-call `stream_text` would
    /// open one stream per token, and frontends key rendering on the stream id per segment.
    #[host(await, name = "_core_stream_text")]
    fn stream_text<'a>(
        &'a self,
        opts: StreamTextOptions,
    ) -> crate::BoxFuture<'a, Result<Box<dyn TextStreamWriter>, RtcError>>;

    /// Open a byte stream and return its writer: one per avatar audio segment.
    #[host(await, name = "_core_stream_bytes")]
    fn stream_bytes<'a>(
        &'a self,
        opts: StreamBytesOptions,
    ) -> crate::BoxFuture<'a, Result<Box<dyn ByteStreamWriter>, RtcError>> {
        Box::pin(async move { Err(RtcError::Unsupported("stream_bytes")) })
    }

    /// `perform_rpc(destination_identity=, method=, payload=)`: `(0, response)` on success, else
    /// `(rtc.RpcError.code, message)`, since callers branch on the code.
    #[host(await, name = "_core_perform_rpc")]
    fn perform_rpc<'a>(
        &'a self,
        destination: ParticipantIdentity,
        method: String,
        payload: String,
    ) -> crate::BoxFuture<'a, Result<(u32, String), RtcError>> {
        Box::pin(async move { Err(RtcError::Unsupported("perform_rpc")) })
    }

    /// `register_rpc_method(method, handler)`: the SDK keeps one handler per method, the last one.
    #[host(call, name = "_core_register_rpc_method")]
    fn register_rpc_method(&self, method: String, handler: RpcHandler) -> Result<(), RtcError> {
        Err(RtcError::Unsupported("register_rpc_method"))
    }

    #[host(call, name = "_core_unregister_rpc_method")]
    fn unregister_rpc_method(&self, method: String) -> Result<(), RtcError> {
        Err(RtcError::Unsupported("unregister_rpc_method"))
    }

    /// `publish_dtmf(code=, digit=)`: a keypad tone to the room's SIP participants.
    #[host(await, name = "_core_publish_dtmf")]
    fn publish_dtmf<'a>(
        &'a self,
        code: u32,
        digit: String,
    ) -> crate::BoxFuture<'a, Result<(), RtcError>> {
        Box::pin(async move { Err(RtcError::Unsupported("publish_dtmf")) })
    }
}

/// An open byte stream. `rtc.ByteStreamWriter`; `close` ends it (the stream's trailer).
#[crate::host(proxy(host = "livekit.agents.voice.room_io.ByteStreamWriter"))]
pub trait ByteStreamWriter: Send + Sync {
    #[host(read_once, name = "_core_id")]
    fn id(&self) -> &str;
    #[host(await, name = "_core_write")]
    fn write<'a>(&'a self, data: &'a [u8]) -> crate::BoxFuture<'a, Result<(), RtcError>>;
    #[host(await, name = "_core_close")]
    fn close(self: Box<Self>) -> crate::BoxFuture<'static, Result<(), RtcError>>;
}

/// An open text stream. `rtc.TextStreamWriter`.
///
/// `close` takes `Box<Self>` so it can be called at most once and the type cannot be used
/// afterwards. Dropping a host stream does not close it, so the close must be explicit
/// and structural rather than remembered -- the reference's own pattern is
/// `writer = self._writer; self._writer = None; await writer.aclose()`.
#[crate::host(proxy(host = "livekit.agents.voice.room_io.TextStreamWriter"))]
pub trait TextStreamWriter: Send + Sync {
    #[host(await, name = "_core_write")]
    fn write<'a>(&'a self, text: &'a str) -> crate::BoxFuture<'a, Result<(), RtcError>>;
    /// Closes with the final header attributes. `rtc`'s `aclose(attributes=...)`.
    #[host(await, name = "_core_close")]
    fn close(
        self: Box<Self>,
        attributes: BTreeMap<String, String>,
    ) -> crate::BoxFuture<'static, Result<(), RtcError>>;
}

/// One published audio track's sink.
///
/// It is separate from [`LocalParticipant`] because "clear the buffer on *this* track" has to be
/// expressible: the avatar path publishes its own source alongside `RoomIO`'s, and an interruption
/// must truncate one and not the other.
#[crate::host(proxy(host = "livekit.agents.voice.room_io.AudioSource"))]
pub trait AudioSource: Send + Sync {
    /// Push one frame. **Awaits when the source's queue is full**, and that is the point: the
    /// `queue_size_ms` bound is what paces the forwarding loop against real playout
    /// (`_forward_audio` awaits `capture_frame` for exactly this reason).
    ///
    /// It cannot be synchronous: without the block there is no
    /// backpressure, the forwarding loop runs the whole utterance into an unbounded queue in one
    /// tick, and `clear_queue` would then have nothing left to truncate -- interruption would stop
    /// cutting audio.
    #[host(await, name = "_core_capture")]
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> crate::BoxFuture<'a, Result<(), RtcError>>;

    /// Discard everything queued but not yet played.
    ///
    /// **Synchronous, deliberately**: this is the interruption path and it must not await the
    /// host loop. It is also why the trait exists -- the queue being cleared belongs to one track.
    #[host(call, name = "_core_clear_queue")]
    fn clear_queue(&self) -> Result<(), RtcError>;

    /// Audio pushed but not yet played. `_forward_audio` reads it to account for what a pause
    /// throws away, and `_wait_for_playout` to find where the playhead stopped.
    #[host(call, name = "_core_queued_duration")]
    fn queued_duration(&self) -> Result<std::time::Duration, RtcError>;

    /// Resolve when the queue drains -- immediately after a [`clear_queue`](Self::clear_queue).
    ///
    /// See [`PlaybackInfo`]: the return value is what the reference reconstructs by hand from
    /// three counters and a race.
    #[host(await, name = "_core_wait_for_playout")]
    fn wait_for_playout<'a>(&'a self) -> crate::BoxFuture<'a, Result<PlaybackInfo, RtcError>>;

    /// The same window [`wait_for_playout`](Self::wait_for_playout) reports, read **now** and
    /// without consuming it: `played` is what has passed the speaker so far (audio still queued
    /// is in neither figure), `discarded` what a `clear_queue` threw away.
    ///
    /// Synchronous: the output pump reads it at a pause and at a interruption, mid-segment, where
    /// `wait_for_playout` has not resolved and cannot. It is the one origin of a playback position
    /// -- nothing above the source reconstructs one from its own counters.
    ///
    /// Defaulted so a host source that has not implemented the read still compiles; the default
    /// reports nothing played, which under-reports every mid-segment position. Implement it.
    #[host(call, name = "_core_playback_info")]
    fn playback_info(&self) -> Result<PlaybackInfo, RtcError>;

    /// `rtc.AudioSource.aclose()`. Idempotent.
    #[host(await, name = "_core_close")]
    fn close<'a>(&'a self) -> crate::BoxFuture<'a, Result<(), RtcError>>;
}

/// Someone else in the room.
#[crate::host(proxy(host = "livekit.agents.voice.room_io.RemoteParticipant"))]
pub trait RemoteParticipant: Send + Sync {
    #[host(read_once, name = "_core_sid")]
    fn sid(&self) -> &str;

    #[host(read_once, name = "_core_identity")]
    fn identity(&self) -> &ParticipantIdentity;

    /// `RoomOptions.participant_kinds` filters auto-linking on this.
    #[host(read_once, name = "_core_kind")]
    fn kind(&self) -> ParticipantKind;

    /// Whether the participant is active (`ParticipantState.PARTICIPANT_STATE_ACTIVE`): connected
    /// and ready to exchange media.
    #[host(call, name = "_core_active")]
    fn active(&self) -> Result<bool, RtcError>;

    /// Read for `ATTRIBUTE_PUBLISH_ON_BEHALF`: a participant publishing on the agent's behalf (an
    /// avatar worker) must not be linked as the user. Returned by value because the host's map is
    /// mutable and a borrow across the FFI would alias it.
    #[host(call, name = "_core_attributes")]
    fn attributes(&self) -> Result<BTreeMap<String, String>, RtcError>;

    #[host(call, name = "_core_track_publications")]
    fn track_publications(&self) -> Result<Vec<Arc<dyn TrackPublication>>, RtcError>;
}

/// A published track, local or remote.
///
/// One trait over `rtc.LocalTrackPublication` and `rtc.RemoteTrackPublication`, as in the SDK
/// where both derive from `TrackPublication`. The two side-specific methods default to
/// [`RtcError::Unsupported`] so a shim implements only its own half; see that variant for why this
/// is one trait and not two.
#[crate::host(proxy(host = "livekit.agents.voice.room_io.TrackPublication"))]
pub trait TrackPublication: Send + Sync {
    #[host(read_once, name = "_core_sid")]
    fn sid(&self) -> &TrackSid;
    #[host(read_once, name = "_core_name")]
    fn name(&self) -> &str;
    #[host(read_once, name = "_core_source")]
    fn source(&self) -> TrackSource;

    /// `rtc.TrackKind`'s number (`TRACK_KIND_AUDIO`, `TRACK_KIND_VIDEO`); a host without it reads
    /// the kind off the source.
    #[host(read_once, name = "_core_kind")]
    fn kind(&self) -> i32;

    /// Whether the track is subscribed and its media can be read. On a remote publication this is
    /// the reference's `publication.track is not None` check.
    #[host(call, name = "_core_is_subscribed")]
    fn is_subscribed(&self) -> Result<bool, RtcError>;

    /// `TF_PRECONNECT_BUFFER` here is what gates the pre-connect audio path (`_input.py`); the
    /// rest of the set has no reader today.
    #[host(read_once, name = "_core_audio_features")]
    fn audio_features(&self) -> Vec<AudioTrackFeature>;

    /// Remote only.
    #[host(await, name = "_core_set_subscribed")]
    fn set_subscribed<'a>(
        &'a self,
        subscribed: bool,
    ) -> crate::BoxFuture<'a, Result<(), RtcError>> {
        Box::pin(async move { Err(RtcError::Unsupported("set_subscribed")) })
    }

    /// Local only: resolve once a remote peer has subscribed.
    ///
    /// `_ParticipantAudioOutput` gates `capture_frame` on this so the first syllable of the
    /// greeting is not published into a room with nobody listening.
    #[host(await, name = "_core_wait_for_subscription")]
    fn wait_for_subscription<'a>(&'a self) -> crate::BoxFuture<'a, Result<(), RtcError>> {
        Box::pin(async move { Err(RtcError::Unsupported("wait_for_subscription")) })
    }

    /// Open a stream of the subscribed audio into `frames`, which ends when the track goes away
    /// (the sender dropped). `dyn TrackPublication::audio_stream` is the reader's form.
    ///
    /// Drop-oldest: audio-in is lossy, so a stalled STT plugin loses the oldest second
    /// rather than growing the process or blocking a host callback that holds the GIL.
    #[host(call, name = "_core_open_audio_stream")]
    fn open_audio_stream(
        &self,
        opts: AudioStreamOptions,
    ) -> Result<Arc<dyn AudioInputStream>, RtcError>;

    /// Open a stream of the subscribed video into `frames` -- **as side-table keys, never
    /// pixels** -- which ends when the track goes away, exactly as the audio one does.
    ///
    /// No options, because `_ParticipantVideoInputStream._create_stream` passes none:
    /// `rtc.VideoStream.from_track(track=track)` and nothing else -- a `format` or a
    /// `capacity` parameter here would be a knob no caller in the reference turns.
    ///
    /// **Required rather than defaulted**, unlike the two side-specific methods above. A default
    /// body could only drop the sender, which is a camera input that reads nothing and says
    /// nothing, on a leg whose entire failure mode is silence. A shim that has not written this
    /// must fail to compile.
    #[host(call, name = "_core_open_video_stream")]
    fn open_video_stream(&self) -> Result<Box<dyn StreamSource<VideoFrameRef>>, RtcError>;
}

/// One SDK audio stream, including the processor state shared by its buffered head and live frames.
#[crate::host(proxy(host = "livekit.agents.voice.room_io._AudioStream"))]
pub trait AudioInputStream: Send + Sync {
    #[host(call, name = "_core_frames")]
    fn frames(&self) -> Result<Box<dyn StreamSource<AudioFrame>>, RtcError>;

    #[host(await, name = "_core_process_audio")]
    fn process_audio<'a>(
        &'a self,
        frames: Vec<AudioFrame>,
    ) -> crate::BoxFuture<'a, Result<Vec<AudioFrame>, RtcError>> {
        Box::pin(async move { Ok(frames) })
    }
}
