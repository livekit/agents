#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "tests drive tokio's sync primitives directly"
)]
use super::*;
use lk_agents_abi::agent::resources::RunResources as _;
use lk_agents_abi::agent::run::{RunOutcome, RunResult, RunSettled, RunUpdate, StreamEvent};
use lk_agents_abi::agent::stream::{RunStream as _, StreamEvents as _};
use lk_agents_abi::{Closed, Ts};
use lk_agents_core::run_stream::projection::Envelope;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{Notify, mpsc};

struct State {
    reading: Notify,
    closing: Notify,
    release: Notify,
    closes: AtomicUsize,
    held: bool,
    deliveries: Mutex<Vec<TextSessionDelivery>>,
    usage: lk_agents_net::signal::Producer<Usage>,
}
struct Fake {
    state: Arc<State>,
    messages: mpsc::Receiver<Message>,
}
impl Input for Fake {
    fn next(
        &mut self,
    ) -> BoxFuture<'_, Result<Option<Message>, lk_agents_abi::errors::ProviderError>> {
        Box::pin(async move {
            self.state.reading.notify_one();
            Ok(self.messages.recv().await)
        })
    }
    fn usage(&self) -> lk_agents_net::signal::Cell<Usage> {
        self.state.usage.cell()
    }
    fn release(&self, delivery: TextSessionDelivery) -> Result<(), Invalid> {
        self.state.deliveries.lock().push(delivery);
        Ok(())
    }
    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async move {
            self.state.closes.fetch_add(1, Ordering::SeqCst);
            self.state.closing.notify_one();
            if self.state.held {
                self.state.release.notified().await;
            }
            Closed::Clean
        })
    }
}
async fn owner(held: bool) -> (RunStream, Arc<RunResources>, Arc<State>, mpsc::Sender<Message>) {
    let env = ProcessEnv::current_runtime();
    let resources =
        Arc::new(RunResources::new(&env, None, None, None, "agent".into(), vec![]).unwrap());
    resources.open().unwrap().await.unwrap();
    let stream = RunStream::new(&env);
    stream.enter().unwrap();
    let state = Arc::new(State {
        reading: Notify::new(),
        closing: Notify::new(),
        release: Notify::new(),
        closes: AtomicUsize::new(0),
        held,
        deliveries: Mutex::default(),
        usage: lk_agents_net::signal::signal(Usage::default()).0,
    });
    let (send, messages) = mpsc::channel(8);
    stream
        .bind(resources.clone(), || Ok(Box::new(Fake { state: state.clone(), messages })))
        .unwrap();
    (stream, resources, state, send)
}
fn finished() -> Message {
    Message::Terminal(StreamOutcome::Run(RunUpdate::Finished(Box::new(RunSettled {
        result: RunResult {
            outcome: RunOutcome::NoOutput,
            interrupted: false,
            new_messages: vec![],
            all_messages: vec![],
            usage: Usage::default(),
            verbatim: None,
            pending_call: None,
            input_requirements: Vec::new(),
        },
        output: None,
    }))))
}
fn text(delta: &str) -> Message {
    Message::Event(Envelope {
        sequence: None,
        agent_name: "agent".into(),
        invocation_id: "run".into(),
        frame_id: None,
        event: StreamEvent::TextDelta { message_id: "message".into(), delta: delta.into() },
    })
}

#[tokio::test]
async fn cancellation_wakes_a_blocked_reader_and_waits_for_owned_source_close() {
    let (stream, resources, state, _send) = owner(true).await;
    let events = stream.events().unwrap();
    assert!(events.begin_read().unwrap());
    let reading = tokio::spawn(async move {
        let result = events.recv().await;
        events.end_read(false);
        result
    });
    state.reading.notified().await;
    stream.cancel();
    stream.cancel();
    state.closing.notified().await;
    assert!(!reading.is_finished());
    assert!(!stream.done());
    assert!(!resources.closed());
    state.release.notify_one();
    assert!(matches!(reading.await.unwrap(), Err(StreamError::Cancelled)));
    stream.aclose().await.unwrap();
    assert!(resources.closed());
    assert_eq!(state.closes.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn abandoning_a_result_waiter_releases_admission_without_cancelling_execution() {
    let (stream, resources, state, send) = owner(false).await;
    let first = stream.begin_result().unwrap();
    let waiting = tokio::spawn(stream.result(first));
    state.reading.notified().await;
    waiting.abort();
    let _ = waiting.await;
    stream.end_result(first, false);
    assert!(!stream.done());
    let second = stream.begin_result().unwrap();
    send.send(finished()).await.unwrap();
    assert!(matches!(
        stream.result(second).await.unwrap(),
        StreamOutcome::Run(RunUpdate::Finished(_))
    ));
    stream.end_result(second, true);
    assert!(!stream.cancelled());
    assert!(resources.closed());
}

#[tokio::test]
async fn closing_an_unstarted_cursor_allows_result_draining_and_keeps_one_view() {
    let (stream, _resources, _state, send) = owner(false).await;
    let events = stream.events().unwrap();
    events.aclose().await.unwrap();
    assert!(matches!(stream.events(), Err(StreamRefused::IteratorClaimed)));
    let consumer = stream.begin_result().unwrap();
    send.send(finished()).await.unwrap();
    stream.result(consumer).await.unwrap();
    stream.end_result(consumer, true);
}

#[tokio::test]
async fn internally_drained_text_is_not_acknowledged_as_presented() {
    let (stream, _resources, state, send) = owner(false).await;
    send.send(text("visible")).await.unwrap();
    let events = stream.events().unwrap();
    events.begin_read().unwrap();
    events.recv().await.unwrap().unwrap();
    events.end_read(true);
    events.aclose().await.unwrap();
    send.send(text(" hidden")).await.unwrap();
    send.send(Message::Release("run".into())).await.unwrap();
    send.send(finished()).await.unwrap();
    let consumer = stream.begin_result().unwrap();
    stream.result(consumer).await.unwrap();
    stream.end_result(consumer, true);
    let deliveries = state.deliveries.lock();
    assert_eq!(deliveries.len(), 1);
    assert_eq!(deliveries[0].messages.get("message").unwrap(), "visible");
}

#[tokio::test]
async fn a_second_cursor_read_is_refused_until_language_delivery_is_finished() {
    let (stream, _resources, _state, send) = owner(false).await;
    send.send(text("visible")).await.unwrap();
    let events = stream.events().unwrap();
    events.begin_read().unwrap();
    events.recv().await.unwrap().unwrap();
    assert!(matches!(
        events.begin_read(),
        Err(StreamError::Refused(StreamRefused::ConsumerActive))
    ));
    events.end_read(true);
    events.aclose().await.unwrap();
    stream.aclose().await.unwrap();
}

#[tokio::test]
async fn a_native_read_not_received_by_the_language_is_retried_and_acknowledged_once() {
    let (stream, _resources, state, send) = owner(false).await;
    send.send(text("visible")).await.unwrap();
    let events = stream.events().unwrap();
    events.begin_read().unwrap();
    let first = events.recv().await.unwrap().unwrap();
    assert!(matches!(
        events.aclose().await,
        Err(StreamError::Refused(StreamRefused::ConsumerActive))
    ));
    // The native future finished but its language waiter was cancelled before delivery.
    events.end_read(false);
    assert!(state.deliveries.lock().is_empty());
    events.begin_read().unwrap();
    let retried = events.recv().await.unwrap().unwrap();
    assert_eq!(first, retried);
    events.end_read(true);
    events.aclose().await.unwrap();
    send.send(Message::Release("run".into())).await.unwrap();
    send.send(finished()).await.unwrap();
    let consumer = stream.begin_result().unwrap();
    stream.result(consumer).await.unwrap();
    stream.end_result(consumer, true);
    assert_eq!(state.deliveries.lock()[0].messages.get("message").unwrap(), "visible");
}

// Catches: a close waiter parked forever on a writer the stream keeps when its settlement task is
// dropped (a cancel or runtime shutdown) before it records the outcome.
#[tokio::test]
async fn a_close_waiter_resolves_when_the_settlement_is_dropped() {
    let (stream, _resources, state, _send) = owner(true).await;
    let closing = stream.aclose();
    state.closing.notified().await;
    stream.0.tasks.cancel_all();
    let closed = tokio::time::timeout(std::time::Duration::from_secs(5), closing).await;
    state.release.notify_one();
    assert!(closed.is_ok(), "the close waiter outlived the settlement");
}

/// A toolset whose authored close never returns.
struct HungClose(Arc<dyn lk_agents_abi::llm::toolsets::Lifetime>);
impl lk_agents_abi::llm::toolsets::ToolsetHooks for HungClose {
    fn lifetime(&self) -> &Arc<dyn lk_agents_abi::llm::toolsets::Lifetime> {
        &self.0
    }
    fn setup(
        &self,
        _: &lk_agents_abi::telemetry::Cx,
        _: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn close(
        &self,
        _: &lk_agents_abi::telemetry::Cx,
    ) -> BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        Box::pin(std::future::pending())
    }
    fn children(
        &self,
    ) -> Result<
        Option<Arc<dyn lk_agents_abi::llm::toolsets::LeaseGroup>>,
        lk_agents_abi::errors::HostError,
    > {
        Ok(None)
    }
}

// Catches: a run stream's close waiting forever on its resources when a toolset's close hangs.
#[tokio::test(start_paused = true)]
async fn a_hung_resource_close_is_bounded_by_the_teardown_deadline() {
    use lk_agents_abi::llm::toolsets::ToolsetLifetime as _;
    let env = ProcessEnv::current_runtime();
    let lifetime = Arc::new(crate::toolsets::lifetime::Lifetime::new(&env));
    let hooks = vec![Arc::new(HungClose(lifetime)) as _];
    let resources =
        Arc::new(RunResources::new(&env, None, None, None, "agent".into(), hooks).unwrap());
    resources.open().unwrap().await.unwrap();
    let stream = RunStream::new(&env);
    stream.enter().unwrap();
    let (_send, messages) = mpsc::channel(8);
    let state = Arc::new(State {
        reading: Notify::new(),
        closing: Notify::new(),
        release: Notify::new(),
        closes: AtomicUsize::new(0),
        held: false,
        deliveries: Mutex::default(),
        usage: lk_agents_net::signal::signal(Usage::default()).0,
    });
    stream.bind(resources, || Ok(Box::new(Fake { state, messages }))).unwrap();
    let limit = lk_agents_abi::HOST_TEARDOWN_TIMEOUT * 3;
    let closed = tokio::time::timeout(limit, stream.aclose()).await;
    assert!(matches!(closed, Ok(Err(_))), "the close outlived its deadline: {closed:?}");
}
