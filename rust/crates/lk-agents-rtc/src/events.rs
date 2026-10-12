//! Room observations, incoming data streams, and event fan-out.

use std::fmt;
use std::sync::{Arc, Mutex};

use lk_agents_abi::{AudioFrame, LossyTx, MediaRef, Pushed};
use tokio::sync::mpsc;

use crate::traits::{RemoteParticipant, TrackPublication};
use crate::{ConnectionState, DisconnectReason, ParticipantIdentity, StreamInfo, VideoFrameRef};

/// Unbounded on purpose: `broadcast` drops the oldest on lag, and a missed `track_subscribed`
/// waits forever. Control events are rare; only media is lossy.
pub type Receiver<T> = mpsc::UnboundedReceiver<T>;

/// One of the eight.
///
/// Handles rather than identifiers in the payloads: §1.5 chose an object graph over a flat session
/// so that "which participant, which publication" and "this publication is gone" are expressible.
/// An event carrying only a `TrackSid` would send every consumer back to the room to look the
/// handle up, and that lookup races the very disconnection the event reports.
#[derive(Clone, lk_agents_abi::Vocabulary)]
pub enum RoomEvent {
    /// A remote track is subscribed and its frames can be read.
    /// `_input.py` reacts by opening an `audio_stream`.
    TrackSubscribed {
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    },
    /// A remote participant published a track (not necessarily subscribed).
    /// The transcription outputs use this to learn the user's microphone `sid`.
    TrackPublished {
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    },
    /// A remote track went away. `_input.py` closes its stream and falls back to the
    /// participant's next available track.
    TrackUnpublished {
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    },
    /// The agent's own track was published; carries the `sid` the agent's transcription is
    /// attached to.
    LocalTrackPublished {
        publication: Arc<dyn TrackPublication>,
    },
    ParticipantConnected {
        participant: Arc<dyn RemoteParticipant>,
    },
    /// Carries the reason **on the event**.
    ///
    /// The reference reads `participant.disconnect_reason` off the participant object at handler
    /// time (`room_io.RoomIO._on_participant_connected`), which only works because Python holds the SDK's own mutable
    /// object. A handle whose `disconnect_reason` is `None` until some later mutation would be a
    /// race across the FFI, so the reason travels with the event that reports it.
    ParticipantDisconnected {
        participant: Arc<dyn RemoteParticipant>,
        reason: DisconnectReason,
    },
    ConnectionStateChanged {
        state: ConnectionState,
    },
    /// `rtc.Room`'s `disconnected` event. Named `close` in §1.5's list and in
    /// `warm_transfer.WarmTransferTask._dial_human_agent`, which is the only reader.
    Close {
        reason: DisconnectReason,
    },
}

// Hand-written so `RemoteParticipant`/`TrackPublication` need not be `Debug`.
impl fmt::Debug for RoomEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoomEvent::TrackSubscribed { participant, publication } => f
                .debug_struct("TrackSubscribed")
                .field("participant", participant.identity())
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::TrackPublished { participant, publication } => f
                .debug_struct("TrackPublished")
                .field("participant", participant.identity())
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::TrackUnpublished { participant, publication } => f
                .debug_struct("TrackUnpublished")
                .field("participant", participant.identity())
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::LocalTrackPublished { publication } => f
                .debug_struct("LocalTrackPublished")
                .field("publication", publication.sid())
                .finish(),
            RoomEvent::ParticipantConnected { participant } => f
                .debug_struct("ParticipantConnected")
                .field("participant", participant.identity())
                .finish(),
            RoomEvent::ParticipantDisconnected { participant, reason } => f
                .debug_struct("ParticipantDisconnected")
                .field("participant", participant.identity())
                .field("reason", reason)
                .finish(),
            RoomEvent::ConnectionStateChanged { state } => {
                f.debug_struct("ConnectionStateChanged").field("state", state).finish()
            }
            RoomEvent::Close { reason } => f.debug_struct("Close").field("reason", reason).finish(),
        }
    }
}

/// One value per stream, carrying its own chunk channel: several can be open at once.
#[derive(Debug)]
pub struct ByteStream {
    pub info: StreamInfo,
    pub sender: ParticipantIdentity,
    pub chunks: Receiver<Vec<u8>>,
}

#[derive(Debug)]
pub struct TextStream {
    pub info: StreamInfo,
    pub sender: ParticipantIdentity,
    pub chunks: Receiver<String>,
}

impl TextStream {
    /// `rtc.TextStreamReader.read_all()`: the chat-input path wants the whole message.
    pub async fn read_all(&mut self) -> String {
        let mut out = String::new();
        while let Some(chunk) = self.chunks.recv().await {
            out.push_str(&chunk);
        }
        out
    }
}

/// Shared by every [`RtcRoom`](crate::RtcRoom): the reference registers four independent
/// `room.on(...)` handlers. Closed subscribers are pruned on the next [`emit`](Self::emit).
#[derive(Default)]
pub struct RoomEventBus {
    subscribers: Mutex<Vec<mpsc::UnboundedSender<RoomEvent>>>,
}

/// A host room's side: `_core_rtc.Room` holds one and emits the SDK's `room.on(..)` events into it,
/// its participants and publications wrapped in the classes that answer the core's traits.
#[lk_agents_abi::host(object(cell = "arc"))]
impl RoomEventBus {
    #[host(new)]
    pub fn new() -> Self {
        RoomEventBus::default()
    }

    #[host(method)]
    pub fn track_subscribed(&self, participant: Arc<dyn RemoteParticipant>, publication: Arc<dyn TrackPublication>) {
        self.emit(RoomEvent::TrackSubscribed { participant, publication });
    }

    #[host(method)]
    pub fn track_published(&self, participant: Arc<dyn RemoteParticipant>, publication: Arc<dyn TrackPublication>) {
        self.emit(RoomEvent::TrackPublished { participant, publication });
    }

    #[host(method)]
    pub fn track_unpublished(&self, participant: Arc<dyn RemoteParticipant>, publication: Arc<dyn TrackPublication>) {
        self.emit(RoomEvent::TrackUnpublished { participant, publication });
    }

    #[host(method)]
    pub fn local_track_published(&self, publication: Arc<dyn TrackPublication>) {
        self.emit(RoomEvent::LocalTrackPublished { publication });
    }

    #[host(method)]
    pub fn participant_connected(&self, participant: Arc<dyn RemoteParticipant>) {
        self.emit(RoomEvent::ParticipantConnected { participant });
    }

    /// The reason is read inside the host's handler and travels on the event; read later, it races
    /// the SDK's own mutation of the participant.
    #[host(method)]
    pub fn participant_disconnected(&self, participant: Arc<dyn RemoteParticipant>, reason: DisconnectReason) {
        self.emit(RoomEvent::ParticipantDisconnected { participant, reason });
    }

    #[host(method)]
    pub fn connection_state_changed(&self, state: ConnectionState) {
        self.emit(RoomEvent::ConnectionStateChanged { state });
    }

    /// `rtc.Room`'s `disconnected(reason)`: §1.5 calls it `close`.
    #[host(method)]
    pub fn disconnected(&self, reason: DisconnectReason) {
        self.emit(RoomEvent::Close { reason });
    }
}

impl RoomEventBus {
    /// Events emitted before this call are not replayed, as with `room.on`.
    pub fn subscribe(&self) -> Receiver<RoomEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.subscribers.lock().unwrap().push(tx);
        rx
    }

    /// Never blocks: it runs in a synchronous host callback holding the GIL.
    pub fn emit(&self, event: RoomEvent) -> usize {
        let mut subs = self.subscribers.lock().unwrap();
        subs.retain(|tx| tx.send(event.clone()).is_ok());
        subs.len()
    }

    /// Dropped receivers are only noticed on send, so this can over-report.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------------------------
// what a host pushes inbound streams and media into
// ---------------------------------------------------------------------------------------------

/// Where a host room delivers the byte streams opened on a registered topic
/// ([`RtcRoom::register_byte_stream_handler`](crate::RtcRoom::register_byte_stream_handler)).
#[lk_agents_abi::class]
#[derive(Clone)]
pub struct ByteStreamSender(pub(crate) mpsc::UnboundedSender<ByteStream>);

impl ByteStreamSender {
    pub fn new(tx: mpsc::UnboundedSender<ByteStream>) -> Self {
        ByteStreamSender(tx)
    }
}

#[lk_agents_abi::host(object)]
impl ByteStreamSender {
    /// One stream opened by `sender`: its chunks go into what this returns, in order. `None` once
    /// nobody reads the topic any more.
    #[host(method)]
    pub fn open(&self, info: StreamInfo, sender: ParticipantIdentity) -> Option<ByteChunks> {
        let (tx, chunks) = mpsc::unbounded_channel();
        self.0.send(ByteStream { info, sender, chunks }).ok()?;
        Some(ByteChunks(Mutex::new(Some(tx))))
    }
}

/// One inbound byte stream's chunks. The stream ends at [`close`](Self::close), or when the host
/// drops this.
#[lk_agents_abi::class]
pub struct ByteChunks(Mutex<Option<mpsc::UnboundedSender<Vec<u8>>>>);

#[lk_agents_abi::host(object)]
impl ByteChunks {
    /// `False` once the stream's reader is gone: stop reading the host stream.
    #[host(method)]
    pub fn push(&self, chunk: &[u8]) -> bool {
        self.0.lock().unwrap().as_ref().is_some_and(|tx| tx.send(chunk.to_vec()).is_ok())
    }

    #[host(method)]
    pub fn close(&self) {
        self.0.lock().unwrap().take();
    }
}

/// Where a host room delivers the text streams opened on a registered topic.
#[lk_agents_abi::class]
#[derive(Clone)]
pub struct TextStreamSender(pub(crate) mpsc::UnboundedSender<TextStream>);

impl TextStreamSender {
    pub fn new(tx: mpsc::UnboundedSender<TextStream>) -> Self {
        TextStreamSender(tx)
    }
}

#[lk_agents_abi::host(object)]
impl TextStreamSender {
    /// One stream opened by `sender`; `None` once nobody reads the topic any more.
    #[host(method)]
    pub fn open(&self, info: StreamInfo, sender: ParticipantIdentity) -> Option<TextChunks> {
        let (tx, chunks) = mpsc::unbounded_channel();
        self.0.send(TextStream { info, sender, chunks }).ok()?;
        Some(TextChunks(Mutex::new(Some(tx))))
    }
}

/// One inbound text stream's chunks; ends at [`close`](Self::close) or when the host drops this.
#[lk_agents_abi::class]
pub struct TextChunks(Mutex<Option<mpsc::UnboundedSender<String>>>);

#[lk_agents_abi::host(object)]
impl TextChunks {
    /// `False` once the stream's reader is gone.
    #[host(method)]
    pub fn push(&self, chunk: String) -> bool {
        self.0.lock().unwrap().as_ref().is_some_and(|tx| tx.send(chunk).is_ok())
    }

    #[host(method)]
    pub fn close(&self) {
        self.0.lock().unwrap().take();
    }
}

/// Where a host track's audio stream delivers its frames, drop-oldest (§1.2d): the push half of a
/// [`BoundedLossy`](lk_agents_abi::BoundedLossy). The stream ends at [`close`](Self::close) or when
/// the host drops this.
#[lk_agents_abi::class]
pub struct AudioFrameSender(Mutex<Option<LossyTx<AudioFrame>>>);

impl AudioFrameSender {
    pub fn new(tx: LossyTx<AudioFrame>) -> Self {
        AudioFrameSender(Mutex::new(Some(tx)))
    }
}

#[lk_agents_abi::host(object)]
impl AudioFrameSender {
    /// Queue one frame, evicting the oldest if the queue is full. `False` once the reader is gone:
    /// close the host stream.
    #[host(method)]
    pub fn push(&self, frame: AudioFrame) -> bool {
        self.0.lock().unwrap().as_ref().is_some_and(|tx| tx.push(frame) != Pushed::Closed)
    }

    #[host(method)]
    pub fn close(&self) {
        self.0.lock().unwrap().take();
    }
}

/// Where a host track's video stream delivers its frames, as side-table keys (the pixels never
/// cross); drop-oldest, like the audio one.
#[lk_agents_abi::class]
pub struct VideoFrameSender(Mutex<Option<LossyTx<VideoFrameRef>>>);

impl VideoFrameSender {
    pub fn new(tx: LossyTx<VideoFrameRef>) -> Self {
        VideoFrameSender(Mutex::new(Some(tx)))
    }
}

#[lk_agents_abi::host(object)]
impl VideoFrameSender {
    /// Queue one host `rtc.VideoFrame` (interned; only its key crosses) with its size and capture
    /// stamp. `False` once the reader is gone.
    #[host(method)]
    pub fn push(&self, frame: MediaRef, width: u32, height: u32, timestamp_us: i64) -> bool {
        let frame = VideoFrameRef { media: frame, width, height, timestamp_us };
        self.0.lock().unwrap().as_ref().is_some_and(|tx| tx.push(frame) != Pushed::Closed)
    }

    #[host(method)]
    pub fn close(&self) {
        self.0.lock().unwrap().take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: `emit` reporting a delivery count that does not match reality, and dropped
    // receivers never being pruned. The count is what a shim would use to notice it is forwarding
    // events nobody reads; without pruning the fan-out grows for the life of the session.
    #[test]
    fn emit_reports_how_many_subscribers_received_it() {
        let bus = RoomEventBus::new();
        assert_eq!(bus.emit(RoomEvent::Close { reason: DisconnectReason::Unknown }), 0);

        let a = bus.subscribe();
        let b = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);
        assert_eq!(bus.emit(RoomEvent::Close { reason: DisconnectReason::Unknown }), 2);

        drop(b);
        assert_eq!(bus.emit(RoomEvent::Close { reason: DisconnectReason::Unknown }), 1);
        assert_eq!(bus.subscriber_count(), 1);
        drop(a);
    }
}
