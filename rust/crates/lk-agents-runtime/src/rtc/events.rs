//! Bounded room observations and admitted SDK streams, using the shared source lifecycle.

use parking_lot::Mutex;
use std::sync::Arc;

use async_trait::async_trait;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::voice::rtc::capabilities::RtcRoom;
use lk_agents_abi::voice::rtc::ports::{
    self, ByteStream, ByteStreamIngress, Receiver, RpcCallback, TextStream, TextStreamIngress,
};
use lk_agents_abi::voice::rtc::{
    ParticipantIdentity, RoomEvent, RoomEventSink, RtcError, StreamInfo,
};
use lk_agents_abi::{Closed, StreamSource, Ts};
use lk_agents_voice::room_io::ingress::{MAX_INCOMING_STREAMS, StreamBudget};
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
        let mut subscribers = self.subscribers.lock();
        subscribers.retain(|subscriber| !subscriber.events.is_closed());
        subscribers.push(Subscriber { events, failure });
        Box::new(EventReceiver { events: receiver, failure: outcome, ended: false })
    }

    /// Overflow terminates this subscription explicitly; callbacks never block under the GIL.
    pub fn emit(&self, event: RoomEvent) -> usize {
        let mut subscribers = self.subscribers.lock();
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
        self.subscribers.lock().iter().filter(|subscriber| !subscriber.events.is_closed()).count()
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

impl RoomEventBus {
    /// A new fan-out as the port a room's SDK callbacks emit into.
    pub fn port() -> Arc<dyn RoomEventSink> {
        Arc::new(Self::new())
    }
}

impl lk_agents_abi::voice::rtc::events::RoomEventBus for RoomEventBus {
    fn new() -> Self {
        Self::default()
    }
}

impl lk_agents_abi::voice::rtc::events::RtcRuntime for RoomEventBus {}

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
    pub fn channel() -> (Arc<dyn ByteStreamIngress>, Receiver<ByteStream>) {
        let (streams, receiver) = mpsc::channel(MAX_INCOMING_STREAMS);
        let sender = Self { streams, admission: Arc::new(Semaphore::new(MAX_INCOMING_STREAMS)) };
        (Arc::new(sender), Box::new(StreamReceiver { streams: receiver }))
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
    pub fn channel() -> (Arc<dyn TextStreamIngress>, Receiver<TextStream>) {
        let (streams, receiver) = mpsc::channel(MAX_INCOMING_STREAMS);
        let sender = Self { streams, admission: Arc::new(Semaphore::new(MAX_INCOMING_STREAMS)) };
        (Arc::new(sender), Box::new(StreamReceiver { streams: receiver }))
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

/// `_core.ByteStreamSender`.
impl ports::ByteStreamSender for ByteStreamSender {
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Receiver<Vec<u8>>,
    ) -> bool {
        ByteStreamIngress::open(self, info, sender, chunks)
    }
}

/// `_core.TextStreamSender`.
impl ports::TextStreamSender for TextStreamSender {
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Receiver<String>,
    ) -> bool {
        TextStreamIngress::open(self, info, sender, chunks)
    }
}

/// An RPC method's handler: `(caller_identity, payload)` to the response payload.
pub struct RpcHandler(Box<dyn Fn(&str, &str) -> String + Send + Sync>);

impl RpcHandler {
    pub fn new(callback: impl Fn(&str, &str) -> String + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self(Box::new(callback)))
    }
}

impl RpcCallback for RpcHandler {
    fn call(&self, caller_identity: &str, payload: &str) -> String {
        (self.0)(caller_identity, payload)
    }
}

/// `_core.RpcHandler`.
impl ports::RpcHandler for RpcHandler {
    fn respond(&self, caller_identity: String, payload: String) -> String {
        (self.0)(&caller_identity, &payload)
    }
}

/// Route every byte stream opened on `topic` into a fresh admitted receiver.
pub fn byte_streams(room: &dyn RtcRoom, topic: &str) -> Result<Receiver<ByteStream>, RtcError> {
    let (sender, receiver) = ByteStreamSender::channel();
    room.register_byte_stream_handler(topic, sender)?;
    Ok(receiver)
}

/// Route every text stream opened on `topic` into a fresh admitted receiver.
pub fn text_streams(room: &dyn RtcRoom, topic: &str) -> Result<Receiver<TextStream>, RtcError> {
    let (sender, receiver) = TextStreamSender::channel();
    room.register_text_stream_handler(topic, sender)?;
    Ok(receiver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::voice::rtc::{ConnectionState, DisconnectReason};
    use lk_agents_voice::room_io::ingress::{MAX_STREAM_BYTES, MAX_STREAM_CHUNK_BYTES};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Chunks {
        values: std::vec::IntoIter<Result<Vec<u8>, ProviderError>>,
        closed: Arc<AtomicUsize>,
    }

    impl StreamSource<Vec<u8>> for Chunks {
        fn next<'a>(
            &'a mut self,
        ) -> lk_agents_abi::BoxFuture<'a, Result<Option<Vec<u8>>, ProviderError>> {
            Box::pin(async move { self.values.next().transpose() })
        }
        fn close(self: Box<Self>, _: Ts) -> lk_agents_abi::BoxFuture<'static, Closed> {
            Box::pin(async move {
                self.closed.fetch_add(1, Ordering::SeqCst);
                Closed::Clean
            })
        }
    }

    fn chunks(
        values: Vec<Result<Vec<u8>, ProviderError>>,
        closed: &Arc<AtomicUsize>,
    ) -> Receiver<Vec<u8>> {
        Box::new(Chunks { values: values.into_iter(), closed: closed.clone() })
    }

    #[tokio::test]
    async fn incoming_stream_admission_includes_dequeued_readers_until_close() {
        let closed = Arc::new(AtomicUsize::new(0));
        let (sender, mut streams) = ByteStreamSender::channel();
        for _ in 0..MAX_INCOMING_STREAMS {
            assert!(sender.open(StreamInfo::default(), "user".into(), chunks(vec![], &closed)));
        }
        let active = streams.next().await.unwrap().unwrap();
        assert!(!sender.open(StreamInfo::default(), "user".into(), chunks(vec![], &closed)));
        assert_eq!(active.close(Ts::ZERO).await, Closed::Clean);
        assert!(sender.open(StreamInfo::default(), "user".into(), chunks(vec![], &closed)));
        assert_eq!(streams.close(Ts::ZERO).await, Closed::Clean);
        assert_eq!(closed.load(Ordering::SeqCst), MAX_INCOMING_STREAMS + 1);
    }

    #[tokio::test]
    async fn incoming_stream_byte_limits_fail_once_and_cleanup_the_original_reader() {
        for values in [
            vec![Ok(vec![0; MAX_STREAM_CHUNK_BYTES + 1])],
            (0..=MAX_STREAM_BYTES / MAX_STREAM_CHUNK_BYTES)
                .map(|_| Ok(vec![0; MAX_STREAM_CHUNK_BYTES]))
                .collect(),
            vec![Err(capacity("SDK reader", 1))],
        ] {
            let closed = Arc::new(AtomicUsize::new(0));
            let (sender, mut streams) = ByteStreamSender::channel();
            assert!(sender.open(StreamInfo::default(), "user".into(), chunks(values, &closed)));
            let mut active = streams.next().await.unwrap().unwrap();
            loop {
                match active.chunks.next().await {
                    Ok(Some(_)) => continue,
                    Ok(None) => panic!("a refused or failed stream must not end successfully"),
                    Err(_) => break,
                }
            }
            assert!(active.chunks.next().await.unwrap().is_none());
            assert_eq!(active.close(Ts::ZERO).await, Closed::Clean);
            assert_eq!(closed.load(Ordering::SeqCst), 1);
            assert_eq!(streams.close(Ts::ZERO).await, Closed::Clean);
        }
    }

    #[tokio::test]
    async fn a_stalled_room_subscription_receives_overflow_after_accepted_events() {
        let bus = RoomEventBus::new();
        let mut subscriber = bus.subscribe();
        for _ in 0..EVENT_CAPACITY {
            assert_eq!(
                bus.emit(RoomEvent::ConnectionStateChanged { state: ConnectionState::Connected }),
                1
            );
        }
        assert_eq!(bus.emit(RoomEvent::Close { reason: DisconnectReason::ClientInitiated }), 0);
        for _ in 0..EVENT_CAPACITY {
            assert!(matches!(
                subscriber.next().await.unwrap(),
                Some(RoomEvent::ConnectionStateChanged { .. })
            ));
        }
        assert!(subscriber.next().await.is_err());
        assert!(subscriber.next().await.unwrap().is_none());
        assert_eq!(subscriber.close(Ts::ZERO).await, Closed::Clean);
    }
}
