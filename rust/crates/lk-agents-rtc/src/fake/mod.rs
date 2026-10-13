//! In-memory RTC for deterministic tests: records operations and models playout on an injected clock.

use crate::events::{
    ByteStream, ByteStreamSender, Receiver, RoomEvent, RoomEventBus, TextStream, TextStreamSender,
};
use crate::traits::{
    AudioSource, LocalParticipant, RemoteParticipant, RtcHost, RtcRoom, TextStreamWriter,
    TrackPublication,
};
use crate::{
    ConnectionState, DisconnectReason, ParticipantIdentity, ParticipantKind, StreamInfo,
    StreamTextOptions, TrackPublishOptions, TrackSid, TrackSource, Transcription,
};
use crate::{Result, RtcError};
use async_trait::async_trait;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::{Clock, VirtualClock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::sync::mpsc;

mod audio_source;
#[cfg(test)]
mod tests;
mod track;

pub use audio_source::{FakeAudioSource, FrameSynth};
pub use track::{FakeTrackPublication, StreamRequests, subscribe_local};

pub struct FakeRtc {
    clock: Arc<dyn Clock>,
    /// `Some` only from [`with_virtual_clock`](FakeRtc::with_virtual_clock); `advance` needs it.
    virtual_clock: Option<Arc<VirtualClock>>,
    state: Mutex<HostState>,
}

#[derive(Default)]
struct HostState {
    rooms: Vec<Arc<FakeRoom>>,
    connects: Vec<ConnectCall>,
    next_connect_error: Option<RtcError>,
}

impl FakeRtc {
    /// Share the session's clock, so playout and the core's timers run on one timeline.
    pub fn new(clock: Arc<dyn Clock>) -> Arc<Self> {
        Arc::new(FakeRtc { clock, virtual_clock: None, state: Mutex::new(HostState::default()) })
    }

    pub fn with_virtual_clock() -> (Arc<Self>, Arc<VirtualClock>) {
        let clock = Arc::new(VirtualClock::new());
        let rtc = Arc::new(FakeRtc {
            clock: clock.clone(),
            virtual_clock: Some(clock.clone()),
            state: Mutex::new(HostState::default()),
        });
        (rtc, clock)
    }

    /// A room with no RTC at all, for console/daemon sessions where the CLI owns the audio. The
    /// clock must be the wall clock the session's timers run on.
    pub fn offline(name: &str, clock: Arc<dyn Clock>) -> (Arc<Self>, Arc<FakeRoom>) {
        let rtc = Self::new(clock);
        let room = rtc.room(name);
        (rtc, room)
    }

    /// Get or create a connected room; idempotent by name.
    pub fn room(&self, name: &str) -> Arc<FakeRoom> {
        let mut state = self.state.lock().unwrap();
        if let Some(existing) = state.rooms.iter().find(|r| r.name == name) {
            return existing.clone();
        }
        let room = FakeRoom::new(name, self.clock.clone());
        state.rooms.push(room.clone());
        room
    }

    pub fn rooms(&self) -> Vec<Arc<FakeRoom>> {
        self.state.lock().unwrap().rooms.clone()
    }

    pub fn connect_calls(&self) -> Vec<ConnectCall> {
        self.state.lock().unwrap().connects.clone()
    }

    /// Fail the next `connect` only.
    pub fn fail_next_connect(&self, err: RtcError) {
        self.state.lock().unwrap().next_connect_error = Some(err);
    }

    /// Panics unless built by [`with_virtual_clock`](FakeRtc::with_virtual_clock); a shared clock
    /// is advanced through that clock, then [`settle`](Self::settle)d.
    pub fn advance(&self, by: Duration) {
        let clock = self
            .virtual_clock
            .as_ref()
            .expect("FakeRtc::advance needs FakeRtc::with_virtual_clock(); use settle() otherwise");
        clock.advance(by);
        self.settle();
    }

    /// Only needed when something else moved the clock: observations settle themselves, parked
    /// waiters cannot.
    pub fn settle(&self) {
        for room in self.rooms() {
            room.settle();
        }
    }
}

#[async_trait]
impl RtcHost for FakeRtc {
    async fn connect(&self, url: &str, token: &str) -> Result<Arc<dyn RtcRoom>> {
        let mut state = self.state.lock().unwrap();
        if let Some(err) = state.next_connect_error.take() {
            return Err(err);
        }
        let name = format!("fake-room-{}", state.rooms.len() + 1);
        let room = FakeRoom::new(&name, self.clock.clone());
        state.rooms.push(room.clone());
        state.connects.push(ConnectCall {
            url: url.to_owned(),
            token: token.to_owned(),
            room: name,
        });
        Ok(room)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectCall {
    pub url: String,
    pub token: String,
    pub room: String,
}

/// Connected from the moment it exists, as `ctx.connect()` leaves it.
pub struct FakeRoom {
    name: String,
    bus: Arc<RoomEventBus>,
    local: Arc<FakeLocalParticipant>,
    local_dyn: Arc<dyn LocalParticipant>,
    /// On by default so unrelated tests need not call `set_subscribed(true)`.
    auto_subscribe: Arc<AtomicBool>,
    state: Mutex<RoomState>,
}

struct RoomState {
    connection: ConnectionState,
    /// A closed room emits nothing more; a later state change is not a reconnect.
    closed: bool,
    participants: Vec<Arc<FakeRemoteParticipant>>,
    byte_handlers: HashMap<String, ByteStreamSender>,
    text_handlers: HashMap<String, TextStreamSender>,
}

impl Default for RoomState {
    // Connected: `ctx.connect()` hands Rust an open room.
    fn default() -> Self {
        RoomState {
            connection: ConnectionState::Connected,
            closed: false,
            participants: Vec::new(),
            byte_handlers: HashMap::new(),
            text_handlers: HashMap::new(),
        }
    }
}

impl FakeRoom {
    fn new(name: &str, clock: Arc<dyn Clock>) -> Arc<Self> {
        let bus = Arc::new(RoomEventBus::new());
        let auto_subscribe = Arc::new(AtomicBool::new(true));
        let local = Arc::new(FakeLocalParticipant::new(clock, bus.clone(), auto_subscribe.clone()));
        Arc::new(FakeRoom {
            name: name.to_owned(),
            local_dyn: local.clone(),
            local,
            bus,
            auto_subscribe,
            state: Mutex::new(RoomState::default()),
        })
    }

    pub fn local_participant(&self) -> Arc<FakeLocalParticipant> {
        self.local.clone()
    }

    pub fn set_auto_subscribe(&self, on: bool) {
        self.auto_subscribe.store(on, Ordering::Relaxed);
    }

    /// A no-op after [`close`](Self::close): `disconnected` is the room's last event.
    pub fn set_connection_state(&self, state: ConnectionState) {
        {
            let mut s = self.state.lock().unwrap();
            if s.closed {
                return;
            }
            s.connection = state;
        }
        self.bus.emit(RoomEvent::ConnectionStateChanged { state });
    }

    /// A duplicate identity drops the older participant first (`DuplicateIdentity`), as the
    /// server does; `RoomIO` re-links on that order.
    pub fn add_participant(
        &self,
        identity: impl Into<ParticipantIdentity>,
        kind: ParticipantKind,
    ) -> Arc<FakeRemoteParticipant> {
        let identity = identity.into();
        self.remove_participant(&identity, DisconnectReason::DuplicateIdentity);
        let p = FakeRemoteParticipant::new(
            identity,
            kind,
            self.bus.clone(),
            self.auto_subscribe.clone(),
        );
        self.state.lock().unwrap().participants.push(p.clone());
        self.bus.emit(RoomEvent::ParticipantConnected { participant: p.clone() });
        p
    }

    /// Tracks are unpublished before `participant_disconnected`, as the server orders it.
    /// Returns `false` if they were not in the room, so a double-disconnect emits once.
    pub fn remove_participant(
        &self,
        identity: &ParticipantIdentity,
        reason: DisconnectReason,
    ) -> bool {
        let removed = {
            let mut state = self.state.lock().unwrap();
            match state.participants.iter().position(|p| &p.identity == identity) {
                Some(i) => state.participants.remove(i),
                None => return false,
            }
        };
        let sids: Vec<TrackSid> =
            removed.state.lock().unwrap().publications.iter().map(|p| p.sid.clone()).collect();
        for sid in &sids {
            removed.unpublish_track(sid);
        }
        self.bus.emit(RoomEvent::ParticipantDisconnected { participant: removed, reason });
        true
    }

    pub fn fake_participant(&self, id: &ParticipantIdentity) -> Option<Arc<FakeRemoteParticipant>> {
        self.state.lock().unwrap().participants.iter().find(|p| &p.identity == id).cloned()
    }

    pub fn close(&self, reason: DisconnectReason) {
        let participants = {
            let mut state = self.state.lock().unwrap();
            state.connection = ConnectionState::Disconnected;
            state.closed = true;
            std::mem::take(&mut state.participants)
        };
        for p in &participants {
            p.close_streams();
        }
        self.local.close_streams();
        self.bus.emit(RoomEvent::ConnectionStateChanged { state: ConnectionState::Disconnected });
        self.bus.emit(RoomEvent::Close { reason });
    }

    pub fn send_text_stream(
        &self,
        sender: impl Into<ParticipantIdentity>,
        info: StreamInfo,
        chunks: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<()> {
        let handler =
            self.state.lock().unwrap().text_handlers.get(&info.topic).cloned().ok_or_else(
                || RtcError::host(format!("no text handler for topic {:?}", info.topic)),
            )?;
        let (tx, rx) = mpsc::unbounded_channel();
        for c in chunks {
            let _ = tx.send(c.into());
        }
        drop(tx);
        handler
            .0
            .send(TextStream { info, sender: sender.into(), chunks: rx })
            .map_err(|_| RtcError::Closed("text stream handler"))
    }

    pub fn send_byte_stream(
        &self,
        sender: impl Into<ParticipantIdentity>,
        info: StreamInfo,
        chunks: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<()> {
        let handler =
            self.state.lock().unwrap().byte_handlers.get(&info.topic).cloned().ok_or_else(
                || RtcError::host(format!("no byte handler for topic {:?}", info.topic)),
            )?;
        let (tx, rx) = mpsc::unbounded_channel();
        for c in chunks {
            let _ = tx.send(c);
        }
        drop(tx);
        handler
            .0
            .send(ByteStream { info, sender: sender.into(), chunks: rx })
            .map_err(|_| RtcError::Closed("byte stream handler"))
    }

    pub fn events(&self) -> Receiver<RoomEvent> {
        self.bus.subscribe()
    }

    pub fn text_streams(&self, topic: &str) -> Result<Receiver<TextStream>> {
        (self as &dyn RtcRoom).text_streams(topic)
    }

    pub fn byte_streams(&self, topic: &str) -> Result<Receiver<ByteStream>> {
        (self as &dyn RtcRoom).byte_streams(topic)
    }

    /// Live event subscribers as of the last emit; how a test checks a `room.off(...)` took effect.
    pub fn event_subscriber_count(&self) -> usize {
        self.bus.subscriber_count()
    }

    pub fn settle(&self) {
        self.local.settle();
    }
}

impl RtcRoom for FakeRoom {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn connection_state(&self) -> ConnectionState {
        self.state.lock().unwrap().connection
    }

    fn local(&self) -> &Arc<dyn LocalParticipant> {
        &self.local_dyn
    }

    fn participant(&self, id: &ParticipantIdentity) -> Option<Arc<dyn RemoteParticipant>> {
        self.state
            .lock()
            .unwrap()
            .participants
            .iter()
            .find(|p| &p.identity == id)
            .map(|p| p.clone() as Arc<dyn RemoteParticipant>)
    }

    fn remote_participants(&self) -> Vec<Arc<dyn RemoteParticipant>> {
        self.state
            .lock()
            .unwrap()
            .participants
            .iter()
            .map(|p| p.clone() as Arc<dyn RemoteParticipant>)
            .collect()
    }

    fn event_bus(&self) -> &Arc<RoomEventBus> {
        &self.bus
    }

    fn register_byte_stream_handler(&self, topic: &str, streams: ByteStreamSender) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.byte_handlers.contains_key(topic) {
            return Err(RtcError::TopicInUse { topic: topic.to_owned() });
        }
        state.byte_handlers.insert(topic.to_owned(), streams);
        Ok(())
    }

    fn unregister_byte_stream_handler(&self, topic: &str) {
        self.state.lock().unwrap().byte_handlers.remove(topic);
    }

    fn register_text_stream_handler(&self, topic: &str, streams: TextStreamSender) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.text_handlers.contains_key(topic) {
            return Err(RtcError::TopicInUse { topic: topic.to_owned() });
        }
        state.text_handlers.insert(topic.to_owned(), streams);
        Ok(())
    }

    fn unregister_text_stream_handler(&self, topic: &str) {
        self.state.lock().unwrap().text_handlers.remove(topic);
    }
}

pub struct FakeLocalParticipant {
    identity: ParticipantIdentity,
    clock: Arc<dyn Clock>,
    bus: Arc<RoomEventBus>,
    auto_subscribe: Arc<AtomicBool>,
    next_sid: AtomicU64,
    state: Mutex<LocalState>,
}

#[derive(Default)]
struct LocalState {
    publications: Vec<Arc<FakeTrackPublication>>,
    sources: Vec<Arc<FakeAudioSource>>,
    attributes: HashMap<String, String>,
    attribute_calls: Vec<HashMap<String, String>>,
    transcriptions: Vec<Transcription>,
    text_streams: Vec<Arc<Mutex<RecordedTextStream>>>,
    fail_with: Option<RtcError>,
}

impl FakeLocalParticipant {
    fn new(clock: Arc<dyn Clock>, bus: Arc<RoomEventBus>, auto_subscribe: Arc<AtomicBool>) -> Self {
        FakeLocalParticipant {
            identity: ParticipantIdentity("agent".to_owned()),
            clock,
            bus,
            auto_subscribe,
            next_sid: AtomicU64::new(1),
            state: Mutex::new(LocalState::default()),
        }
    }

    pub fn attributes(&self) -> HashMap<String, String> {
        self.state.lock().unwrap().attributes.clone()
    }

    /// Each call separately: the reference cancels an in-flight update, so the count matters.
    pub fn attribute_calls(&self) -> Vec<HashMap<String, String>> {
        self.state.lock().unwrap().attribute_calls.clone()
    }

    pub fn publications(&self) -> Vec<Arc<FakeTrackPublication>> {
        self.state.lock().unwrap().publications.clone()
    }

    pub fn sources(&self) -> Vec<Arc<FakeAudioSource>> {
        self.state.lock().unwrap().sources.clone()
    }

    pub fn captured_bytes(&self) -> usize {
        self.sum_bytes(|s| s.captured())
    }

    pub fn played_bytes(&self) -> usize {
        self.sum_bytes(|s| s.played())
    }

    pub fn discarded_bytes(&self) -> usize {
        self.sum_bytes(|s| s.discarded())
    }

    fn sum_bytes(&self, f: impl Fn(&FakeAudioSource) -> Vec<AudioFrame>) -> usize {
        self.sources().iter().flat_map(|s| f(s)).map(|frame| frame.data.len()).sum()
    }

    pub fn transcriptions(&self) -> Vec<Transcription> {
        self.state.lock().unwrap().transcriptions.clone()
    }

    pub fn text_streams(&self) -> Vec<RecordedTextStream> {
        self.state.lock().unwrap().text_streams.iter().map(|s| s.lock().unwrap().clone()).collect()
    }

    /// Fail every subsequent control call.
    pub fn fail_with(&self, err: RtcError) {
        self.state.lock().unwrap().fail_with = Some(err);
    }

    fn check(&self) -> Result<()> {
        match self.state.lock().unwrap().fail_with.clone() {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn settle(&self) {
        let sources = self.state.lock().unwrap().sources.clone();
        for s in sources {
            s.settle();
        }
    }

    fn close_streams(&self) {
        let pubs = self.state.lock().unwrap().publications.clone();
        for p in pubs {
            p.close_streams();
        }
    }
}

#[async_trait]
impl LocalParticipant for FakeLocalParticipant {
    fn identity(&self) -> ParticipantIdentity {
        self.identity.clone()
    }

    fn track_publications(&self) -> Vec<Arc<dyn TrackPublication>> {
        self.state
            .lock()
            .unwrap()
            .publications
            .iter()
            .map(|p| p.clone() as Arc<dyn TrackPublication>)
            .collect()
    }

    async fn publish_audio_track(
        &self,
        opts: TrackPublishOptions,
    ) -> Result<(Arc<dyn AudioSource>, Arc<dyn TrackPublication>)> {
        self.check()?;
        let sid = TrackSid(format!("TR_local_{}", self.next_sid.fetch_add(1, Ordering::Relaxed)));
        let source = Arc::new(FakeAudioSource::new(self.clock.clone(), &opts));
        let mut publication = FakeTrackPublication::new(
            sid,
            opts.source,
            self.bus.clone(),
            None,
            self.auto_subscribe.load(Ordering::Relaxed),
        );
        Arc::get_mut(&mut publication).unwrap().name = opts.name;
        {
            let mut state = self.state.lock().unwrap();
            state.publications.push(publication.clone());
            state.sources.push(source.clone());
        }
        self.bus.emit(RoomEvent::LocalTrackPublished { publication: publication.clone() });
        Ok((source, publication))
    }

    async fn unpublish_track(&self, sid: &TrackSid) -> Result<()> {
        self.check()?;
        let mut state = self.state.lock().unwrap();
        let index = state
            .publications
            .iter()
            .position(|p| p.sid() == sid)
            .ok_or_else(|| RtcError::TrackGone(sid.clone()))?;
        state.publications.remove(index).close_streams();
        Ok(())
    }

    async fn set_attributes(&self, attrs: HashMap<String, String>) -> Result<()> {
        self.check()?;
        let mut state = self.state.lock().unwrap();
        state.attributes.extend(attrs.iter().map(|(k, v)| (k.clone(), v.clone())));
        state.attribute_calls.push(attrs);
        Ok(())
    }

    async fn publish_transcription(&self, transcription: Transcription) -> Result<()> {
        self.check()?;
        self.state.lock().unwrap().transcriptions.push(transcription);
        Ok(())
    }

    async fn stream_text(&self, opts: StreamTextOptions) -> Result<Box<dyn TextStreamWriter>> {
        self.check()?;
        let recorded = Arc::new(Mutex::new(RecordedTextStream {
            opts,
            chunks: Vec::new(),
            closed: false,
            close_attributes: HashMap::new(),
        }));
        self.state.lock().unwrap().text_streams.push(recorded.clone());
        Ok(Box::new(FakeTextStreamWriter { recorded }))
    }
}

struct FakeTextStreamWriter {
    recorded: Arc<Mutex<RecordedTextStream>>,
}

#[async_trait]
impl TextStreamWriter for FakeTextStreamWriter {
    async fn write(&self, text: &str) -> Result<()> {
        // No closed-check: `close` takes `Box<Self>`, so writing after it does not compile.
        self.recorded.lock().unwrap().chunks.push(text.to_owned());
        Ok(())
    }

    async fn close(self: Box<Self>, attributes: HashMap<String, String>) -> Result<()> {
        let mut r = self.recorded.lock().unwrap();
        r.closed = true;
        r.close_attributes = attributes;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedTextStream {
    pub opts: StreamTextOptions,
    pub chunks: Vec<String>,
    pub closed: bool,
    /// The header written at close: `lk.transcription_final`, `lk.expression`, …
    pub close_attributes: HashMap<String, String>,
}

impl RecordedTextStream {
    pub fn text(&self) -> String {
        self.chunks.concat()
    }
}

pub struct FakeRemoteParticipant {
    identity: ParticipantIdentity,
    kind: ParticipantKind,
    bus: Arc<RoomEventBus>,
    auto_subscribe: Arc<AtomicBool>,
    me: Weak<FakeRemoteParticipant>,
    state: Mutex<RemoteState>,
}

#[derive(Default)]
struct RemoteState {
    attributes: HashMap<String, String>,
    publications: Vec<Arc<FakeTrackPublication>>,
}

impl FakeRemoteParticipant {
    fn new(
        identity: ParticipantIdentity,
        kind: ParticipantKind,
        bus: Arc<RoomEventBus>,
        auto_subscribe: Arc<AtomicBool>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| FakeRemoteParticipant {
            identity,
            kind,
            bus,
            auto_subscribe,
            me: me.clone(),
            state: Mutex::new(RemoteState::default()),
        })
    }

    /// `RoomIO` refuses to link a participant whose `lk.publish_on_behalf` is the agent's identity.
    pub fn set_attribute(&self, key: impl Into<String>, value: impl Into<String>) {
        self.state.lock().unwrap().attributes.insert(key.into(), value.into());
    }

    /// Emits `track_published`, and `track_subscribed` unless auto-subscribe is off.
    pub fn publish_track(
        &self,
        sid: impl Into<TrackSid>,
        source: TrackSource,
    ) -> Arc<FakeTrackPublication> {
        let auto = self.auto_subscribe.load(Ordering::Relaxed);
        let owner = self.me.clone();
        let publication =
            FakeTrackPublication::new(sid.into(), source, self.bus.clone(), Some(owner), false);
        self.state.lock().unwrap().publications.push(publication.clone());

        let me = self.me.upgrade().expect("participant dropped while publishing");
        self.bus
            .emit(RoomEvent::TrackPublished { participant: me, publication: publication.clone() });
        if auto {
            publication.mark_subscribed(true);
        }
        publication
    }

    pub fn publish_microphone(&self, sid: impl Into<TrackSid>) -> Arc<FakeTrackPublication> {
        self.publish_track(sid, TrackSource::Microphone)
    }

    pub fn publications(&self) -> Vec<Arc<FakeTrackPublication>> {
        self.state.lock().unwrap().publications.clone()
    }

    /// Also ends the frame stream, so a reader sees `None` rather than hanging.
    pub fn unpublish_track(&self, sid: &TrackSid) -> bool {
        let publication = {
            let mut state = self.state.lock().unwrap();
            match state.publications.iter().position(|p| &p.sid == sid) {
                Some(i) => state.publications.remove(i),
                None => return false,
            }
        };
        publication.close_streams();
        let me = self.me.upgrade().expect("participant dropped while unpublishing");
        self.bus.emit(RoomEvent::TrackUnpublished { participant: me, publication });
        true
    }

    fn close_streams(&self) {
        let pubs = self.state.lock().unwrap().publications.clone();
        for p in pubs {
            p.close_streams();
        }
    }
}

impl RemoteParticipant for FakeRemoteParticipant {
    fn identity(&self) -> &ParticipantIdentity {
        &self.identity
    }

    fn kind(&self) -> ParticipantKind {
        self.kind
    }

    fn attributes(&self) -> HashMap<String, String> {
        self.state.lock().unwrap().attributes.clone()
    }

    fn track_publications(&self) -> Vec<Arc<dyn TrackPublication>> {
        self.state
            .lock()
            .unwrap()
            .publications
            .iter()
            .map(|p| p.clone() as Arc<dyn TrackPublication>)
            .collect()
    }
}

#[doc(hidden)]
pub fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime").block_on(f)
}
