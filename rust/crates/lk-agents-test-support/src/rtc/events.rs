//! Modeled room event and stream admission for the supplied-clock RTC test support.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::voice::rtc::capabilities::{RemoteParticipant, TrackPublication};
use lk_agents_abi::voice::rtc::ports::{
    self, ByteStream, ByteStreamIngress, Receiver, TextStream, TextStreamIngress,
};
use lk_agents_voice::room_io::ingress::{MAX_INCOMING_STREAMS, StreamBudget};
use lk_agents_abi::voice::rtc::{
    ConnectionState, DisconnectReason, ParticipantIdentity, RoomEvent, RoomEventSink, StreamInfo,
};
use lk_agents_abi::{Closed, StreamSource, Ts};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, watch};

const EVENT_CAPACITY: usize = 128;

fn capacity(resource: &str, limit: usize) -> ProviderError {
    ProviderError::Local(ApiError::new(format!("RTC {resource} exceeds limit {limit}")))
}

struct Subscriber {
    events: mpsc::Sender<RoomEvent>,
    failure: watch::Sender<Option<ProviderError>>,
}

struct EventReceiver {
    events: mpsc::Receiver<RoomEvent>,
    failure: watch::Receiver<Option<ProviderError>>,
    ended: bool,
}

impl StreamSource<RoomEvent> for EventReceiver {
    fn next<'a>(
        &'a mut self,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<RoomEvent>, ProviderError>> {
        Box::pin(async move {
            if self.ended {
                return Ok(None);
            }
            if let Some(event) = self.events.recv().await {
                return Ok(Some(event));
            }
            self.ended = true;
            self.failure.borrow().clone().map_or(Ok(None), Err)
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { Closed::Clean })
    }
}

#[derive(Default)]
pub struct RoomEventBus {
    subscribers: Mutex<Vec<Subscriber>>,
}

impl RoomEventBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn subscribe(&self) -> Receiver<RoomEvent> {
        let (events, receiver) = mpsc::channel(EVENT_CAPACITY);
        let (failure, outcome) = watch::channel(None);
        let mut subscribers = self.subscribers.lock().unwrap();
        subscribers.retain(|subscriber| !subscriber.events.is_closed());
        subscribers.push(Subscriber { events, failure });
        Box::new(EventReceiver { events: receiver, failure: outcome, ended: false })
    }

    /// Overflow terminates this subscription explicitly; callbacks never block under the GIL.
    pub fn emit(&self, event: RoomEvent) -> usize {
        let mut subscribers = self.subscribers.lock().unwrap();
        subscribers.retain(|subscriber| match subscriber.events.try_send(event.clone()) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Closed(_)) => false,
            Err(mpsc::error::TrySendError::Full(_)) => {
                subscriber.failure.send_replace(Some(capacity("room events", EVENT_CAPACITY)));
                false
            }
        });
        subscribers.len()
    }

    pub fn subscriber_count(&self) -> usize {
        self.subscribers
            .lock()
            .unwrap()
            .iter()
            .filter(|subscriber| !subscriber.events.is_closed())
            .count()
    }
}

impl RoomEventSink for RoomEventBus {
    fn subscribe(&self) -> Receiver<RoomEvent> {
        self.subscribe()
    }
    fn emit(&self, event: RoomEvent) -> usize {
        self.emit(event)
    }
    fn subscriber_count(&self) -> usize {
        self.subscriber_count()
    }
}

struct AdmittedSource<T> {
    source: Receiver<T>,
    budget: StreamBudget,
    ended: bool,
    size: fn(&T) -> usize,
    _admission: OwnedSemaphorePermit,
}

impl<T: Send + 'static> StreamSource<T> for AdmittedSource<T> {
    fn next<'a>(&'a mut self) -> lk_agents_abi::BoxFuture<'a, Result<Option<T>, ProviderError>> {
        Box::pin(async move {
            if self.ended {
                return Ok(None);
            }
            let chunk = match self.source.next().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => {
                    self.ended = true;
                    return Ok(None);
                }
                Err(error) => {
                    self.ended = true;
                    return Err(error);
                }
            };
            if let Err(error) = self.budget.admit((self.size)(&chunk)) {
                self.ended = true;
                return Err(ProviderError::Local(ApiError::new(error.to_string())));
            }
            Ok(Some(chunk))
        })
    }

    fn cancel_safe(&self) -> bool {
        self.source.cancel_safe()
    }

    fn close(self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move { self.source.close(deadline).await })
    }
}

#[async_trait]
trait Incoming: Send + 'static {
    async fn close(self, deadline: Ts) -> Closed;
}

#[async_trait]
impl Incoming for ByteStream {
    async fn close(self, deadline: Ts) -> Closed {
        self.close(deadline).await
    }
}

#[async_trait]
impl Incoming for TextStream {
    async fn close(self, deadline: Ts) -> Closed {
        self.close(deadline).await
    }
}

struct StreamReceiver<T> {
    streams: mpsc::Receiver<T>,
}

impl<T: Incoming> StreamSource<T> for StreamReceiver<T> {
    fn next<'a>(&'a mut self) -> lk_agents_abi::BoxFuture<'a, Result<Option<T>, ProviderError>> {
        Box::pin(async move { Ok(self.streams.recv().await) })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(mut self: Box<Self>, deadline: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.streams.close();
            let mut outcome = Closed::Clean;
            while let Some(stream) = self.streams.recv().await {
                outcome = outcome.and(stream.close(deadline).await);
            }
            outcome
        })
    }
}

pub struct ByteStreamSender {
    streams: mpsc::Sender<ByteStream>,
    admission: Arc<Semaphore>,
}

impl ByteStreamSender {
    pub fn channel() -> (ports::ByteStreamSender, Receiver<ByteStream>) {
        let (streams, receiver) = mpsc::channel(MAX_INCOMING_STREAMS);
        let sender = Self { streams, admission: Arc::new(Semaphore::new(MAX_INCOMING_STREAMS)) };
        (ports::ByteStreamSender(Arc::new(sender)), Box::new(StreamReceiver { streams: receiver }))
    }
}

impl ByteStreamIngress for ByteStreamSender {
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Receiver<Vec<u8>>,
    ) -> bool {
        let Ok(admission) = self.admission.clone().try_acquire_owned() else { return false };
        let Ok(header) = self.streams.try_reserve() else { return false };
        header.send(ByteStream {
            info,
            sender,
            chunks: Box::new(AdmittedSource {
                source: chunks,
                budget: StreamBudget::default(),
                ended: false,
                size: Vec::len,
                _admission: admission,
            }),
        });
        true
    }
}

pub struct TextStreamSender {
    streams: mpsc::Sender<TextStream>,
    admission: Arc<Semaphore>,
}

impl TextStreamSender {
    pub fn channel() -> (ports::TextStreamSender, Receiver<TextStream>) {
        let (streams, receiver) = mpsc::channel(MAX_INCOMING_STREAMS);
        let sender = Self { streams, admission: Arc::new(Semaphore::new(MAX_INCOMING_STREAMS)) };
        (ports::TextStreamSender(Arc::new(sender)), Box::new(StreamReceiver { streams: receiver }))
    }
}

impl TextStreamIngress for TextStreamSender {
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Receiver<String>,
    ) -> bool {
        let Ok(admission) = self.admission.clone().try_acquire_owned() else { return false };
        let Ok(header) = self.streams.try_reserve() else { return false };
        header.send(TextStream {
            info,
            sender,
            chunks: Box::new(AdmittedSource {
                source: chunks,
                budget: StreamBudget::default(),
                ended: false,
                size: String::len,
                _admission: admission,
            }),
        });
        true
    }
}

impl RoomEventBus {
    pub fn track_subscribed(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.emit(RoomEvent::TrackSubscribed { participant, publication });
    }

    pub fn track_unsubscribed(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.emit(RoomEvent::TrackUnsubscribed { participant, publication });
    }

    pub fn track_published(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.emit(RoomEvent::TrackPublished { participant, publication });
    }

    pub fn track_unpublished(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        publication: Arc<dyn TrackPublication>,
    ) {
        self.emit(RoomEvent::TrackUnpublished { participant, publication });
    }

    pub fn local_track_published(&self, publication: Arc<dyn TrackPublication>) {
        self.emit(RoomEvent::LocalTrackPublished { publication });
    }

    pub fn participant_connected(&self, participant: Arc<dyn RemoteParticipant>) {
        self.emit(RoomEvent::ParticipantConnected { participant });
    }

    pub fn participant_active(&self, participant: Arc<dyn RemoteParticipant>) {
        self.emit(RoomEvent::ParticipantActive { participant });
    }

    pub fn participant_attributes_changed(&self, participant: Arc<dyn RemoteParticipant>) {
        self.emit(RoomEvent::ParticipantAttributesChanged { participant });
    }

    /// The reason is read inside the host's handler and travels on the event; read later, it races
    /// the SDK's own mutation of the participant.
    pub fn participant_disconnected(
        &self,
        participant: Arc<dyn RemoteParticipant>,
        reason: DisconnectReason,
    ) {
        self.emit(RoomEvent::ParticipantDisconnected { participant, reason });
    }

    pub fn connection_state_changed(&self, state: ConnectionState) {
        self.emit(RoomEvent::ConnectionStateChanged { state });
    }

    /// `rtc.Room`'s `disconnected(reason)`.
    pub fn disconnected(&self, reason: DisconnectReason) {
        self.emit(RoomEvent::Close { reason });
    }

    /// `rtc.Room`'s `sip_dtmf_received(ev)`: `ev.digit`, and `ev.participant`'s identity.
    pub fn sip_dtmf_received(&self, digit: String, participant: Option<ParticipantIdentity>) {
        self.emit(RoomEvent::SipDtmf { digit, participant });
    }
}
