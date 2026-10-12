//! Host-provided rooms, participants, tracks, streams and audio sources, as bare `#[host(proxy)]`
//! traits. They are `live`: a getter asks the host every time, since rooms connect after hand-over.

use crate::{
    AudioFrameSender, ByteStream, ByteStreamSender, Receiver, RoomEvent, RoomEventBus, TextStream,
    TextStreamSender, VideoFrameSender,
};
use crate::{
    AudioStreamOptions, AudioTrackFeature, ConnectionState, ParticipantIdentity, ParticipantKind,
    PlaybackInfo, StreamTextOptions, TrackPublishOptions, TrackSid, TrackSource, Transcription,
    VideoFrameRef,
};
use crate::{Result, RtcError};
use async_trait::async_trait;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::BoundedLossy;
use std::collections::HashMap;
use std::sync::Arc;

/// Only `warm_transfer`'s second room; the session's own room arrives already open.
#[async_trait]
pub trait RtcHost: Send + Sync {
    async fn connect(&self, url: &str, token: &str) -> Result<Arc<dyn RtcRoom>>;
}

/// A connected room.
///
/// `connection_state` rather than `isconnected`: the reference's boolean cannot distinguish
/// disconnected from reconnecting, and [`ConnectionState::is_connected`] recovers it exactly.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io._core_rtc.Room", live))]
pub trait RtcRoom: Send + Sync {
    /// Owned and asked every time: a room's name is `""` until it connects (N409).
    fn name(&self) -> String;
    fn connection_state(&self) -> ConnectionState;
    /// Lent, so read once: the agent's participant object outlives its connection (the host's
    /// resolves the SDK's on each call).
    fn local(&self) -> &Arc<dyn LocalParticipant>;

    /// One remote participant by identity, or `None` if they are not in the room.
    fn participant(&self, id: &ParticipantIdentity) -> Option<Arc<dyn RemoteParticipant>>;

    /// Every remote participant.
    ///
    /// Not in §1.5's sketch, and unavoidable: `RoomIO._init_task` replays
    /// `room.remote_participants.values()` through its `participant_connected` handler to catch
    /// anyone who joined before the handler was registered, and `set_participant` scans the same
    /// map. Without it, linking to a participant who was already in the room would depend on an
    /// event that has already fired.
    fn remote_participants(&self) -> Vec<Arc<dyn RemoteParticipant>>;

    /// Where the room emits the eight events. `dyn RtcRoom::events` subscribes to it; a host
    /// room pushes into it from its own handlers.
    fn event_bus(&self) -> &Arc<RoomEventBus>;

    /// Route every stream opened on `topic` into `streams`. Errors if the topic already has a
    /// handler -- the SDK's `ValueError`, which both callers treat as a warn-and-continue.
    /// `dyn RtcRoom::byte_streams` is the reader's form.
    fn register_byte_stream_handler(&self, topic: &str, streams: ByteStreamSender) -> Result<()>;
    /// Idempotent: unregistering a topic with no handler is a no-op, because `aclose` runs on
    /// paths where `register` may never have succeeded.
    fn unregister_byte_stream_handler(&self, topic: &str);
    fn register_text_stream_handler(&self, topic: &str, streams: TextStreamSender) -> Result<()>;
    fn unregister_text_stream_handler(&self, topic: &str);
}

impl dyn RtcRoom + '_ {
    /// A new, independent subscription to the eight events.
    pub fn events(&self) -> Receiver<RoomEvent> {
        self.event_bus().subscribe()
    }

    pub fn byte_streams(&self, topic: &str) -> Result<Receiver<ByteStream>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.register_byte_stream_handler(topic, ByteStreamSender::new(tx))?;
        Ok(rx)
    }

    pub fn text_streams(&self, topic: &str) -> Result<Receiver<TextStream>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.register_text_stream_handler(topic, TextStreamSender::new(tx))?;
        Ok(rx)
    }
}

/// The agent itself in the room.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io._core_rtc.LocalParticipant", live))]
#[async_trait]
pub trait LocalParticipant: Send + Sync {
    /// Needed by three readers, all comparing against it: `_init_task` sets the agent transcript's
    /// participant to it, `_on_participant_connected` skips participants publishing *on behalf of*
    /// it, and `_on_local_track_published` gates on it. Owned and asked every time: before the room
    /// connects it is `""`, which no reader can match (N409).
    fn identity(&self) -> ParticipantIdentity;

    /// `find_micro_track_id` scans these when the agent is its own transcription target.
    fn track_publications(&self) -> Vec<Arc<dyn TrackPublication>>;

    /// Create an `rtc.AudioSource`, wrap it in a `LocalAudioTrack`, publish it.
    ///
    /// Returns both halves because they have different lifetimes and different users: the source
    /// is the data plane (`capture`/`clear_queue`), the publication is identity (`sid`, for the
    /// agent's own transcription) and subscription state.
    async fn publish_audio_track(
        &self,
        opts: TrackPublishOptions,
    ) -> Result<(Arc<dyn AudioSource>, Arc<dyn TrackPublication>)>;

    async fn unpublish_track(&self, _sid: &TrackSid) -> Result<()> {
        Err(RtcError::Unsupported("unpublish_track"))
    }

    /// `ATTRIBUTE_AGENT_STATE` is the only writer; the reference cancels an in-flight call before
    /// issuing the next, so implementations need not serialise.
    async fn set_attributes(&self, attrs: HashMap<String, String>) -> Result<()>;

    /// The deprecated transcription API. Still published, in parallel with the text stream, by
    /// `_ParticipantLegacyTranscriptionOutput`.
    async fn publish_transcription(&self, transcription: Transcription) -> Result<()>;

    /// Open a text stream and return its **writer**.
    ///
    /// §1.5 sketched this as `stream_text(topic, text) -> Result<()>`, which does not fit the
    /// reference: `_ParticipantStreamTranscriptionOutput` opens one writer per transcript segment,
    /// writes every delta into it, and closes it with `lk.transcription_final=true` plus the
    /// segment's expression attributes (`_output.py:_flush_task`). A per-call `stream_text` would
    /// open one stream per token, and frontends key rendering on the stream id per segment.
    async fn stream_text(&self, opts: StreamTextOptions) -> Result<Box<dyn TextStreamWriter>>;
}

/// An open text stream. `rtc.TextStreamWriter`.
///
/// `close` takes `Box<Self>` so it can be called at most once and the type cannot be used
/// afterwards. §1.6 #7: dropping a host stream does not close it, so the close must be explicit
/// and structural rather than remembered -- the reference's own pattern is
/// `writer = self._writer; self._writer = None; await writer.aclose()`.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io._core_rtc.TextStreamWriter"))]
#[async_trait]
pub trait TextStreamWriter: Send + Sync {
    async fn write(&self, text: &str) -> Result<()>;
    /// Closes with the final header attributes. `rtc`'s `aclose(attributes=...)`.
    async fn close(self: Box<Self>, attributes: HashMap<String, String>) -> Result<()>;
}

/// One published audio track's sink. **The load-bearing trait** (§1.5).
///
/// It is separate from [`LocalParticipant`] because "clear the buffer on *this* track" has to be
/// expressible: the avatar path publishes its own source alongside `RoomIO`'s, and an interruption
/// must truncate one and not the other.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io._core_rtc.AudioSource", live))]
#[async_trait]
pub trait AudioSource: Send + Sync {
    /// Push one frame. **Awaits when the source's queue is full**, and that is the point: the
    /// `queue_size_ms` bound is what paces the forwarding loop against real playout
    /// (`_forward_audio` awaits `capture_frame` for exactly this reason).
    ///
    /// §1.5's sketch had this synchronous. It cannot be: without the block there is no
    /// backpressure, the forwarding loop runs the whole utterance into an unbounded queue in one
    /// tick, and `clear_queue` would then have nothing left to truncate -- interruption would stop
    /// cutting audio.
    async fn capture(&self, frame: &AudioFrame) -> Result<()>;

    /// Discard everything queued but not yet played.
    ///
    /// **Synchronous, deliberately** (§1.5): this is the barge-in path and it must not await the
    /// host loop. It is also why the trait exists -- the queue being cleared belongs to one track.
    fn clear_queue(&self);

    /// Audio pushed but not yet played. `_forward_audio` reads it to account for what a pause
    /// throws away, and `_wait_for_playout` to find where the playhead stopped.
    fn queued_duration(&self) -> std::time::Duration;

    /// Resolve when the queue drains -- immediately after a [`clear_queue`](Self::clear_queue).
    ///
    /// See [`PlaybackInfo`]: the return value is what the reference reconstructs by hand from
    /// three counters and a race.
    async fn wait_for_playout(&self) -> PlaybackInfo;

    /// The same window [`wait_for_playout`](Self::wait_for_playout) reports, read **now** and
    /// without consuming it: `played` is what has passed the speaker so far (audio still queued
    /// is in neither figure), `discarded` what a `clear_queue` threw away.
    ///
    /// Synchronous: the output pump reads it at a pause and at a barge-in, mid-segment, where
    /// `wait_for_playout` has not resolved and cannot. It is the one origin of a playback position
    /// -- nothing above the source reconstructs one from its own counters.
    ///
    /// Defaulted so a host source that has not implemented the read still compiles; the default
    /// reports nothing played, which under-reports every mid-segment position. Implement it.
    fn playback_info(&self) -> PlaybackInfo {
        PlaybackInfo::default()
    }

    /// `rtc.AudioSource.aclose()`. Idempotent.
    async fn close(&self) -> Result<()>;
}

/// Someone else in the room.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io._core_rtc.RemoteParticipant", live))]
pub trait RemoteParticipant: Send + Sync {
    fn identity(&self) -> &ParticipantIdentity;

    /// `RoomOptions.participant_kinds` filters auto-linking on this.
    fn kind(&self) -> ParticipantKind;

    /// Read for `ATTRIBUTE_PUBLISH_ON_BEHALF`: a participant publishing on the agent's behalf (an
    /// avatar worker) must not be linked as the user. Returned by value because the host's map is
    /// mutable and a borrow across the FFI would alias it.
    fn attributes(&self) -> HashMap<String, String>;

    fn track_publications(&self) -> Vec<Arc<dyn TrackPublication>>;
}

/// A published track, local or remote.
///
/// One trait over `rtc.LocalTrackPublication` and `rtc.RemoteTrackPublication`, as in the SDK
/// where both derive from `TrackPublication`. The two side-specific methods default to
/// [`RtcError::Unsupported`] so a shim implements only its own half; see that variant for why this
/// is one trait and not two.
#[lk_agents_abi::host(proxy(host = "livekit.agents.voice.room_io._core_rtc.TrackPublication", live))]
#[async_trait]
pub trait TrackPublication: Send + Sync {
    fn sid(&self) -> &TrackSid;
    fn name(&self) -> &str {
        ""
    }
    fn source(&self) -> TrackSource;

    /// Whether the track is subscribed and its media can be read. On a remote publication this is
    /// the reference's `publication.track is not None` check.
    fn is_subscribed(&self) -> bool;

    /// `TF_PRECONNECT_BUFFER` here is what gates the pre-connect audio path (`_input.py`); the
    /// rest of the set has no reader today.
    fn audio_features(&self) -> Vec<AudioTrackFeature> {
        Vec::new()
    }

    /// Remote only.
    async fn set_subscribed(&self, _subscribed: bool) -> Result<()> {
        Err(RtcError::Unsupported("set_subscribed"))
    }

    /// Local only: resolve once a remote peer has subscribed.
    ///
    /// `_ParticipantAudioOutput` gates `capture_frame` on this so the first syllable of the
    /// greeting is not published into a room with nobody listening.
    async fn wait_for_subscription(&self) -> Result<()> {
        Err(RtcError::Unsupported("wait_for_subscription"))
    }

    /// Open a stream of the subscribed audio into `frames`, which ends when the track goes away
    /// (the sender dropped). `dyn TrackPublication::audio_stream` is the reader's form.
    ///
    /// Drop-oldest: §1.2d makes audio-in lossy, so a stalled STT plugin loses the oldest second
    /// rather than growing the process or blocking a host callback that holds the GIL.
    fn open_audio_stream(&self, opts: AudioStreamOptions, frames: AudioFrameSender);

    /// `_apply_audio_processor` (`_input.py:367`): the pre-connect buffer through the frame
    /// processor the last `open_audio_stream` opened with (noise cancellation), unchanged when it
    /// opened with none.
    async fn process_audio(&self, frames: Vec<AudioFrame>) -> Result<Vec<AudioFrame>> {
        Ok(frames)
    }

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
    fn open_video_stream(&self, frames: VideoFrameSender);
}

impl dyn TrackPublication + '_ {
    /// Ends (`recv` returns `None`) when the track goes away.
    pub fn audio_stream(&self, opts: AudioStreamOptions) -> BoundedLossy<AudioFrame> {
        let (tx, rx) = BoundedLossy::audio();
        self.open_audio_stream(opts, AudioFrameSender::new(tx));
        rx
    }

    pub fn video_stream(&self) -> BoundedLossy<VideoFrameRef> {
        let (tx, rx) = BoundedLossy::channel(crate::VIDEO_QUEUE_FRAMES);
        self.open_video_stream(VideoFrameSender::new(tx));
        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four methods a shim must write, and nothing else. Exists to pin the *defaults*: this is
    /// the shape a host publication wrapper takes when it only implements its own half.
    struct MinimalPublication(TrackSid);

    #[async_trait]
    impl TrackPublication for MinimalPublication {
        fn sid(&self) -> &TrackSid {
            &self.0
        }
        fn source(&self) -> TrackSource {
            TrackSource::Microphone
        }
        fn is_subscribed(&self) -> bool {
            true
        }
        fn open_audio_stream(&self, _opts: AudioStreamOptions, _frames: AudioFrameSender) {}
        fn open_video_stream(&self, _frames: VideoFrameSender) {}
    }

    /// Bound every await: a default that parks instead of refusing must fail by name, not hang.
    /// `start_paused` makes tokio fire this the instant the task parks, so it costs no real time.
    async fn eventually<F: std::future::Future>(f: F) -> F::Output {
        tokio::time::timeout(std::time::Duration::from_secs(3600), f)
            .await
            .expect("a default trait body blocked instead of returning")
    }

    // Catches: the side-specific defaults quietly succeeding. A shim that forgets to override
    // `wait_for_subscription` on its *local* publication would then report the track subscribed
    // before anyone had subscribed, and `_ParticipantAudioOutput` would publish the greeting into
    // an empty room -- the exact bug the gate exists to prevent, made silent by an `Ok(())`.
    #[tokio::test(start_paused = true)]
    async fn unimplemented_sides_refuse_rather_than_succeed() {
        let p = MinimalPublication(TrackSid::from("TR_x"));

        assert_eq!(
            eventually(p.set_subscribed(true)).await,
            Err(RtcError::Unsupported("set_subscribed"))
        );
        assert_eq!(
            eventually(p.wait_for_subscription()).await,
            Err(RtcError::Unsupported("wait_for_subscription"))
        );
        assert!(p.audio_features().is_empty(), "no features unless the shim reports some");
    }

    // Catches: a video stream whose sender is kept alive with nobody to push into it. A pump reads
    // this queue in a `while let Some(..)` loop, so a queue that never closes is a task that never
    // ends -- teardown then waits on a camera that is already gone. Dropping the sender is what
    // makes "the track went away" observable at all, and it is the half a shim forgets.
    #[tokio::test(start_paused = true)]
    async fn a_video_stream_with_no_sender_ends_rather_than_parks() {
        let p = MinimalPublication(TrackSid::from("TR_x"));
        let mut stream = (&p as &dyn TrackPublication).video_stream();
        assert!(eventually(stream.recv()).await.is_none());
    }
}
