//! The room the DST runs the runtime's room executors in: the ABI room capabilities in memory,
//! recording what was done and playing out on the run's clock, as the LiveKit SDK behaves.
use lk_agents_abi::voice::rtc::RoomEventSink;

use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::voice::rtc::capabilities::{
    AudioSource, LocalParticipant, RemoteParticipant, RtcRoom, TextStreamWriter, TrackPublication,
};
use lk_agents_abi::voice::rtc::{
    ByteStreamIngress, ConnectionState, DisconnectReason, ParticipantIdentity, ParticipantKind,
    Receiver, RoomEvent, RpcCallback, RtcError, StreamInfo, StreamTextOptions, TextStreamIngress,
    TrackPublishOptions, TrackSid, TrackSource, Transcription,
};
use lk_agents_abi::{AudioFrame, Closed, StreamSource, Ts};
use lk_agents_abi::{Clock, VirtualClock};
use lk_agents_runtime::room_io::bounded_lossy::BoundedLossy;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::sync::mpsc;

type Result<T> = std::result::Result<T, RtcError>;

mod audio_source;
mod track;

pub use audio_source::{FakeAudioSource, FrameSynth};
pub use track::{FakeTrackPublication, StreamRequests, subscribe_local};

pub struct FakeRtc {
    clock: Arc<dyn Clock>,
    /// The runtime's room event fan-out each room's SDK callbacks emit into.
    bus: fn() -> Arc<dyn RoomEventSink>,
    /// `Some` only from [`with_virtual_clock`](FakeRtc::with_virtual_clock); `advance` needs it.
    virtual_clock: Option<Arc<VirtualClock>>,
    rooms: Mutex<Vec<Arc<FakeRoom>>>,
}

impl FakeRtc {
    /// Share the session's clock, so playout and the core's timers run on one timeline.
    pub fn new(clock: Arc<dyn Clock>, bus: fn() -> Arc<dyn RoomEventSink>) -> Arc<Self> {
        Arc::new(FakeRtc { clock, bus, virtual_clock: None, rooms: Mutex::default() })
    }

    pub fn with_virtual_clock(bus: fn() -> Arc<dyn RoomEventSink>) -> (Arc<Self>, Arc<VirtualClock>) {
        let clock = Arc::new(VirtualClock::new());
        let rtc = Arc::new(FakeRtc {
            clock: clock.clone(),
            bus,
            virtual_clock: Some(clock.clone()),
            rooms: Mutex::default(),
        });
        (rtc, clock)
    }

    /// Get or create a connected room; idempotent by name.
    pub fn room(&self, name: &str) -> Arc<FakeRoom> {
        let mut rooms = self.rooms.lock().unwrap();
        if let Some(existing) = rooms.iter().find(|r| r.name == name) {
            return existing.clone();
        }
        let room = FakeRoom::new(name, self.clock.clone(), (self.bus)());
        rooms.push(room.clone());
        room
    }

    pub fn rooms(&self) -> Vec<Arc<FakeRoom>> {
        self.rooms.lock().unwrap().clone()
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

/// Connected from the moment it exists, as `ctx.connect()` leaves it.
pub struct FakeRoom {
    name: String,
    bus: Arc<dyn RoomEventSink>,
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
    next_participant_id: u64,
    byte_handlers: BTreeMap<String, Arc<dyn ByteStreamIngress>>,
    text_handlers: BTreeMap<String, Arc<dyn TextStreamIngress>>,
}

impl Default for RoomState {
    // Connected: `ctx.connect()` hands Rust an open room.
    fn default() -> Self {
        RoomState {
            connection: ConnectionState::Connected,
            closed: false,
            participants: Vec::new(),
            next_participant_id: 1,
            byte_handlers: BTreeMap::new(),
            text_handlers: BTreeMap::new(),
        }
    }
}

impl FakeRoom {
    fn new(name: &str, clock: Arc<dyn Clock>, bus: Arc<dyn RoomEventSink>) -> Arc<Self> {
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

    /// The SDK's resume after a dropped connection: `Reconnecting` back to `Connected`. False
    /// from any other state.
    pub fn reconnected(&self) -> bool {
        {
            let mut s = self.state.lock().unwrap();
            if s.closed || s.connection != ConnectionState::Reconnecting {
                return false;
            }
            s.connection = ConnectionState::Connected;
        }
        self.bus.emit(RoomEvent::ConnectionStateChanged { state: ConnectionState::Connected });
        true
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
        let sid = {
            let mut state = self.state.lock().unwrap();
            let id = state.next_participant_id;
            state.next_participant_id = id.checked_add(1).expect("fake participant ids exhausted");
            format!("PA_{id}")
        };
        let p = FakeRemoteParticipant::new(
            identity,
            sid,
            kind,
            self.bus.clone(),
            self.auto_subscribe.clone(),
        );
        self.state.lock().unwrap().participants.push(p.clone());
        self.bus.emit(RoomEvent::ParticipantConnected { participant: p.clone() });
        // a fake participant is active as it joins
        self.bus.emit(RoomEvent::ParticipantActive { participant: p.clone() });
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

    /// Participant `who` presses `digit` on the keypad; `false` when they are not in the room.
    pub fn press_dtmf(&self, who: &ParticipantIdentity, digit: &str) -> bool {
        if self.fake_participant(who).is_none() {
            return false;
        }
        let participant = Some(who.clone());
        self.bus.emit(RoomEvent::SipDtmf { digit: digit.to_owned(), participant });
        true
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
        let chunks = chunks.into_iter().map(Into::into).collect();
        if handler.open(info, sender.into(), Box::new(FiniteSource(chunks))) {
            Ok(())
        } else {
            Err(RtcError::Closed("text stream handler"))
        }
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
        let chunks = chunks.into_iter().collect();
        if handler.open(info, sender.into(), Box::new(FiniteSource(chunks))) {
            Ok(())
        } else {
            Err(RtcError::Closed("byte stream handler"))
        }
    }

    /// One pending stream; dropping the supplied sender is its trailer.
    pub fn open_byte_stream(
        &self,
        sender: impl Into<ParticipantIdentity>,
        info: StreamInfo,
    ) -> Result<mpsc::Sender<Vec<u8>>> {
        let handler =
            self.state.lock().unwrap().byte_handlers.get(&info.topic).cloned().ok_or_else(
                || RtcError::host(format!("no byte handler for topic {:?}", info.topic)),
            )?;
        let (tx, rx) = mpsc::channel(4);
        if !handler.open(info, sender.into(), Box::new(ChannelSource(rx))) {
            return Err(RtcError::Closed("byte stream handler"));
        }
        Ok(tx)
    }

    pub fn events(&self) -> Receiver<RoomEvent> {
        self.bus.subscribe()
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
    fn name(&self) -> Result<String> {
        Ok(self.name.clone())
    }

    fn connection_state(&self) -> Result<ConnectionState> {
        Ok(self.state.lock().unwrap().connection)
    }

    fn local(&self) -> &Arc<dyn LocalParticipant> {
        &self.local_dyn
    }

    fn participant(&self, id: &ParticipantIdentity) -> Result<Option<Arc<dyn RemoteParticipant>>> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .participants
            .iter()
            .find(|p| &p.identity == id)
            .map(|p| p.clone() as Arc<dyn RemoteParticipant>))
    }

    fn remote_participants(&self) -> Result<Vec<Arc<dyn RemoteParticipant>>> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .participants
            .iter()
            .map(|p| p.clone() as Arc<dyn RemoteParticipant>)
            .collect())
    }

    fn event_bus(&self) -> &Arc<dyn RoomEventSink> {
        &self.bus
    }

    fn register_byte_stream_handler(
        &self,
        topic: &str,
        streams: Arc<dyn ByteStreamIngress>,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.byte_handlers.contains_key(topic) {
            return Err(RtcError::TopicInUse { topic: topic.to_owned() });
        }
        state.byte_handlers.insert(topic.to_owned(), streams);
        Ok(())
    }

    fn unregister_byte_stream_handler(&self, topic: &str) -> Result<()> {
        self.state.lock().unwrap().byte_handlers.remove(topic);
        Ok(())
    }

    fn register_text_stream_handler(
        &self,
        topic: &str,
        streams: Arc<dyn TextStreamIngress>,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.text_handlers.contains_key(topic) {
            return Err(RtcError::TopicInUse { topic: topic.to_owned() });
        }
        state.text_handlers.insert(topic.to_owned(), streams);
        Ok(())
    }

    fn unregister_text_stream_handler(&self, topic: &str) -> Result<()> {
        self.state.lock().unwrap().text_handlers.remove(topic);
        Ok(())
    }
}

pub struct FakeLocalParticipant {
    identity: ParticipantIdentity,
    clock: Arc<dyn Clock>,
    bus: Arc<dyn RoomEventSink>,
    auto_subscribe: Arc<AtomicBool>,
    next_sid: AtomicU64,
    state: Mutex<LocalState>,
    /// `false` holds every byte stream write, as a congested link does.
    byte_writes: tokio::sync::watch::Sender<bool>,
    byte_closes: tokio::sync::watch::Sender<bool>,
    publications_flow: tokio::sync::watch::Sender<bool>,
    /// Everything sent over the data channel, in the order it went out.
    sent: Arc<SentLog>,
}

/// What a participant sent, and how much: a peer relaying it waits on the count.
#[derive(Default)]
struct SentLog {
    items: Mutex<Vec<Sent>>,
    count: tokio::sync::watch::Sender<usize>,
}

impl SentLog {
    fn push(&self, item: Sent) {
        let mut items = self.items.lock().unwrap();
        items.push(item);
        self.count.send_replace(items.len());
    }
}

/// One thing the participant sent over its reliable data channel, which a peer receives in this
/// order: byte streams are named by their index in [`FakeLocalParticipant::byte_streams`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sent {
    StreamOpened(usize),
    StreamChunk(usize, Vec<u8>),
    StreamClosed(usize),
    Rpc { destination: ParticipantIdentity, method: String, payload: String },
}

#[derive(Default)]
struct LocalState {
    byte_stream_id: Option<String>,
    publications: Vec<Arc<FakeTrackPublication>>,
    sources: Vec<Arc<FakeAudioSource>>,
    attributes: BTreeMap<String, String>,
    attribute_calls: Vec<BTreeMap<String, String>>,
    transcriptions: Vec<Transcription>,
    text_streams: Vec<Arc<Mutex<RecordedTextStream>>>,
    byte_streams: Vec<Arc<Mutex<RecordedByteStream>>>,
    rpc_calls: Vec<(ParticipantIdentity, String, String)>,
    /// What the next `perform_rpc` answers; `(0, "")` when unset.
    rpc_reply: Option<(u32, String)>,
    rpc_handlers: BTreeMap<String, Arc<dyn RpcCallback>>,
    dtmf: Vec<(u32, String)>,
    fail_with: Option<RtcError>,
}

impl FakeLocalParticipant {
    fn new(
        clock: Arc<dyn Clock>,
        bus: Arc<dyn RoomEventSink>,
        auto_subscribe: Arc<AtomicBool>,
    ) -> Self {
        FakeLocalParticipant {
            identity: ParticipantIdentity("agent".to_owned()),
            clock,
            bus,
            auto_subscribe,
            next_sid: AtomicU64::new(1),
            state: Mutex::new(LocalState::default()),
            byte_writes: tokio::sync::watch::channel(true).0,
            byte_closes: tokio::sync::watch::channel(true).0,
            publications_flow: tokio::sync::watch::channel(true).0,
            sent: Arc::default(),
        }
    }

    /// Hold a publish response, including across caller cancellation and room teardown.
    pub fn flow_publications(&self, flowing: bool) {
        self.publications_flow.send_replace(flowing);
    }

    /// Hold (`false`) or release byte stream writes: a write in flight awaits, as on a real link.
    pub fn flow_byte_writes(&self, flowing: bool) {
        self.byte_writes.send_replace(flowing);
    }

    /// Supply a transport-selected ID, including reuse across output generations.
    pub fn set_byte_stream_id(&self, id: Option<String>) {
        self.state.lock().unwrap().byte_stream_id = id;
    }

    /// Hold byte stream close acknowledgements while retaining each writer.
    pub fn flow_byte_closes(&self, flowing: bool) {
        self.byte_closes.send_replace(flowing);
    }

    /// What was sent over the data channel from the `from`-th send on, in order.
    pub fn sent_since(&self, from: usize) -> Vec<Sent> {
        let sent = self.sent.items.lock().unwrap();
        sent.get(from..).map(<[Sent]>::to_vec).unwrap_or_default()
    }

    /// How many things were sent, changing with each send.
    pub fn sent_count(&self) -> tokio::sync::watch::Receiver<usize> {
        self.sent.count.subscribe()
    }

    pub fn attributes(&self) -> BTreeMap<String, String> {
        self.state.lock().unwrap().attributes.clone()
    }

    /// Every keypad tone published, as `(code, digit)`.
    pub fn dtmf_published(&self) -> Vec<(u32, String)> {
        self.state.lock().unwrap().dtmf.clone()
    }

    /// Each call separately: the reference cancels an in-flight update, so the count matters.
    pub fn attribute_calls(&self) -> Vec<BTreeMap<String, String>> {
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

    pub fn byte_streams(&self) -> Vec<RecordedByteStream> {
        self.state.lock().unwrap().byte_streams.iter().map(|s| s.lock().unwrap().clone()).collect()
    }

    /// Every `perform_rpc`, as `(destination, method, payload)`.
    pub fn rpc_calls(&self) -> Vec<(ParticipantIdentity, String, String)> {
        self.state.lock().unwrap().rpc_calls.clone()
    }

    /// Answer every later `perform_rpc` with `(code, text)`; code 0 is success.
    pub fn set_rpc_reply(&self, code: u32, text: impl Into<String>) {
        self.state.lock().unwrap().rpc_reply = Some((code, text.into()));
    }

    /// `caller` invokes `method` on the agent; `None` when no handler is registered.
    pub fn invoke_rpc(&self, method: &str, caller: &str, payload: &str) -> Option<String> {
        let handler = self.state.lock().unwrap().rpc_handlers.get(method).cloned()?;
        Some(handler.call(caller, payload))
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

impl LocalParticipant for FakeLocalParticipant {
    fn kind(&self) -> Result<ParticipantKind> {
        self.check()?;
        Ok(ParticipantKind::Agent)
    }

    fn publish_dtmf<'a>(
        &'a self,
        code: u32,
        digit: String,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.state.lock().unwrap().dtmf.push((code, digit));
            Ok(())
        })
    }

    fn identity(&self) -> Result<ParticipantIdentity> {
        self.check()?;
        Ok(self.identity.clone())
    }

    fn track_publications(&self) -> Result<Vec<Arc<dyn TrackPublication>>> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .publications
            .iter()
            .map(|p| p.clone() as Arc<dyn TrackPublication>)
            .collect())
    }

    fn publish_audio_track<'a>(
        &'a self,
        opts: TrackPublishOptions,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(Arc<dyn AudioSource>, Arc<dyn TrackPublication>)>>
    {
        Box::pin(async move {
            self.check()?;
            let sid =
                TrackSid(format!("TR_local_{}", self.next_sid.fetch_add(1, Ordering::Relaxed)));
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
            let _ = self.publications_flow.subscribe().wait_for(|flowing| *flowing).await;
            Ok((source as Arc<dyn AudioSource>, publication as Arc<dyn TrackPublication>))
        })
    }

    fn unpublish_track<'a>(
        &'a self,
        sid: &'a TrackSid,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.check()?;
            let mut state = self.state.lock().unwrap();
            let index = state
                .publications
                .iter()
                .position(|p| p.sid() == sid)
                .ok_or_else(|| RtcError::TrackGone(sid.clone()))?;
            state.publications.remove(index).close_streams();
            Ok(())
        })
    }

    fn set_attributes<'a>(
        &'a self,
        attrs: BTreeMap<String, String>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.check()?;
            let mut state = self.state.lock().unwrap();
            state.attributes.extend(attrs.iter().map(|(k, v)| (k.clone(), v.clone())));
            state.attribute_calls.push(attrs);
            Ok(())
        })
    }

    fn publish_transcription<'a>(
        &'a self,
        transcription: Transcription,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.check()?;
            self.state.lock().unwrap().transcriptions.push(transcription);
            Ok(())
        })
    }

    fn stream_text<'a>(
        &'a self,
        opts: StreamTextOptions,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Box<dyn TextStreamWriter>>> {
        Box::pin(async move {
            self.check()?;
            let recorded = Arc::new(Mutex::new(RecordedTextStream {
                opts,
                chunks: Vec::new(),
                closed: false,
                close_attributes: BTreeMap::new(),
            }));
            self.state.lock().unwrap().text_streams.push(recorded.clone());
            Ok(Box::new(FakeTextStreamWriter { recorded }) as Box<dyn TextStreamWriter>)
        })
    }

    fn stream_bytes<'a>(
        &'a self,
        opts: lk_agents_abi::voice::rtc::StreamBytesOptions,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Box<dyn lk_agents_abi::voice::rtc::ByteStreamWriter>>>
    {
        Box::pin(async move {
            self.check()?;
            let (recorded, index, id) = {
                let mut state = self.state.lock().unwrap();
                let index = state.byte_streams.len();
                let id = state.byte_stream_id.clone().unwrap_or_else(|| format!("BYTES_{index}"));
                let recorded = Arc::new(Mutex::new(RecordedByteStream {
                    id: id.clone(),
                    opts,
                    chunks: Vec::new(),
                    closed: false,
                }));
                state.byte_streams.push(recorded.clone());
                (recorded, index, id)
            };
            self.sent.push(Sent::StreamOpened(index));
            Ok(Box::new(FakeByteStreamWriter {
                id,
                recorded,
                flowing: self.byte_writes.subscribe(),
                closing: self.byte_closes.subscribe(),
                index,
                sent: self.sent.clone(),
            }) as Box<dyn lk_agents_abi::voice::rtc::ByteStreamWriter>)
        })
    }

    fn perform_rpc<'a>(
        &'a self,
        destination: ParticipantIdentity,
        method: String,
        payload: String,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(u32, String)>> {
        Box::pin(async move {
            self.check()?;
            let mut state = self.state.lock().unwrap();
            self.sent.push(Sent::Rpc {
                destination: destination.clone(),
                method: method.clone(),
                payload: payload.clone(),
            });
            state.rpc_calls.push((destination, method, payload));
            Ok(state.rpc_reply.clone().unwrap_or((0, String::new())))
        })
    }

    fn register_rpc_method(&self, method: String, handler: Arc<dyn RpcCallback>) -> Result<()> {
        self.state.lock().unwrap().rpc_handlers.insert(method, handler);
        Ok(())
    }
    fn unregister_rpc_method(&self, method: String) -> Result<()> {
        self.state.lock().unwrap().rpc_handlers.remove(&method);
        Ok(())
    }
}

struct FakeByteStreamWriter {
    id: String,
    recorded: Arc<Mutex<RecordedByteStream>>,
    flowing: tokio::sync::watch::Receiver<bool>,
    closing: tokio::sync::watch::Receiver<bool>,
    index: usize,
    sent: Arc<SentLog>,
}

impl lk_agents_abi::voice::rtc::ByteStreamWriter for FakeByteStreamWriter {
    fn id(&self) -> &str {
        &self.id
    }
    fn write<'a>(&'a self, data: &'a [u8]) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let _ = self.flowing.clone().wait_for(|flowing| *flowing).await;
            self.recorded.lock().unwrap().chunks.push(data.to_vec());
            self.sent.push(Sent::StreamChunk(self.index, data.to_vec()));
            Ok(())
        })
    }

    fn close(self: Box<Self>) -> lk_agents_abi::BoxFuture<'static, Result<()>> {
        Box::pin(async move {
            let _ = self.closing.clone().wait_for(|flowing| *flowing).await;
            self.recorded.lock().unwrap().closed = true;
            self.sent.push(Sent::StreamClosed(self.index));
            Ok(())
        })
    }
}

/// One byte stream the agent opened: its options, what was written, and whether it was closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedByteStream {
    pub id: String,
    pub opts: lk_agents_abi::voice::rtc::StreamBytesOptions,
    pub chunks: Vec<Vec<u8>>,
    pub closed: bool,
}

struct FakeTextStreamWriter {
    recorded: Arc<Mutex<RecordedTextStream>>,
}

impl TextStreamWriter for FakeTextStreamWriter {
    fn write<'a>(&'a self, text: &'a str) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            // No closed-check: `close` takes `Box<Self>`, so writing after it does not compile.
            self.recorded.lock().unwrap().chunks.push(text.to_owned());
            Ok(())
        })
    }

    fn close(
        self: Box<Self>,
        attributes: BTreeMap<String, String>,
    ) -> lk_agents_abi::BoxFuture<'static, Result<()>> {
        Box::pin(async move {
            let mut r = self.recorded.lock().unwrap();
            r.closed = true;
            r.close_attributes = attributes;
            Ok(())
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedTextStream {
    pub opts: StreamTextOptions,
    pub chunks: Vec<String>,
    pub closed: bool,
    /// The header written at close: `lk.transcription_final`, `lk.expression`, …
    pub close_attributes: BTreeMap<String, String>,
}

impl RecordedTextStream {
    pub fn text(&self) -> String {
        self.chunks.concat()
    }
}

pub struct FakeRemoteParticipant {
    identity: ParticipantIdentity,
    sid: String,
    kind: ParticipantKind,
    bus: Arc<dyn RoomEventSink>,
    auto_subscribe: Arc<AtomicBool>,
    me: Weak<FakeRemoteParticipant>,
    state: Mutex<RemoteState>,
}

#[derive(Default)]
struct RemoteState {
    attributes: BTreeMap<String, String>,
    publications: Vec<Arc<FakeTrackPublication>>,
}

impl FakeRemoteParticipant {
    fn new(
        identity: ParticipantIdentity,
        sid: String,
        kind: ParticipantKind,
        bus: Arc<dyn RoomEventSink>,
        auto_subscribe: Arc<AtomicBool>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| FakeRemoteParticipant {
            identity,
            sid,
            kind,
            bus,
            auto_subscribe,
            me: me.clone(),
            state: Mutex::new(RemoteState::default()),
        })
    }

    /// `RoomIO` refuses to link a participant whose `lk.publish_on_behalf` is the agent's identity.
    /// Emits `participant_attributes_changed`.
    pub fn set_attribute(&self, key: impl Into<String>, value: impl Into<String>) {
        self.state.lock().unwrap().attributes.insert(key.into(), value.into());
        let me = self.me.upgrade().expect("participant dropped while changing its attributes");
        self.bus.emit(RoomEvent::ParticipantAttributesChanged { participant: me });
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
        let me = self.me.upgrade().expect("participant dropped while unpublishing");
        // A subscribed track is unsubscribed first, as the SDK reports an unpublish.
        let subscribed =
            publication.is_subscribed().expect("fake subscription is an in-memory read");
        publication.close_streams();
        if subscribed {
            let publication = publication.clone();
            self.bus.emit(RoomEvent::TrackUnsubscribed { participant: me.clone(), publication });
        }
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
    fn sid(&self) -> &str {
        &self.sid
    }

    fn identity(&self) -> &ParticipantIdentity {
        &self.identity
    }

    fn kind(&self) -> ParticipantKind {
        self.kind
    }

    fn active(&self) -> Result<bool> {
        Ok(true)
    }

    fn attributes(&self) -> Result<BTreeMap<String, String>> {
        Ok(self.state.lock().unwrap().attributes.clone())
    }

    fn track_publications(&self) -> Result<Vec<Arc<dyn TrackPublication>>> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .publications
            .iter()
            .map(|p| p.clone() as Arc<dyn TrackPublication>)
            .collect())
    }
}

/// Supplied finite data; no independently running producer or queued task.
struct FiniteSource<T>(VecDeque<T>);

impl<T: Send + 'static> StreamSource<T> for FiniteSource<T> {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<Option<T>, ProviderError>> {
        Box::pin(async move { Ok(self.0.pop_front()) })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

struct ChannelSource<T>(mpsc::Receiver<T>);

impl<T: Send + 'static> StreamSource<T> for ChannelSource<T> {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<Option<T>, ProviderError>> {
        Box::pin(async move { Ok(self.0.recv().await) })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

struct LossySource<T>(BoundedLossy<T>);

impl<T: Send + 'static> StreamSource<T> for LossySource<T> {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, std::result::Result<Option<T>, ProviderError>> {
        Box::pin(async move { Ok(self.0.recv().await) })
    }
    fn cancel_safe(&self) -> bool {
        true
    }
    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

#[cfg(test)]
mod tests;
