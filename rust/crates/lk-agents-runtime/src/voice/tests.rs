//! Voice execution contracts: core effects, correlated failures, queue priority and accepted views.

use super::*;
use crate::voice::session::{DriverBuilder, Journal, SessionExecutor, SessionHandle, Timebase};
use lk_agents_voice::agent_session::*;
use lk_agents_voice::trace::Step;
use parking_lot::Mutex;
use std::future::Future as _;
use std::time::Duration;

type Steps = Arc<Mutex<Vec<Step>>>;

fn builder(name: &str) -> (DriverBuilder, Steps) {
    let time = Timebase::from_virtual(Arc::new(lk_agents_abi::VirtualClock::new()));
    let config = lk_agents_voice::agent_activity::DriverConfig::new(
        Default::default(),
        lk_agents_voice::agent_activity::options::SessionLegs::none(),
    );
    let steps: Steps = Arc::default();
    let observed = steps.clone();
    let spawner = Arc::new(lk_agents_net::task::TokioSpawner::current());
    let mut builder = DriverBuilder::new("agent".into(), time, spawner, config);
    builder.journal = Journal::observed(move |step| observed.lock().push(step.clone()));
    builder.session_id = Some(name.into());
    (builder, steps)
}

pub(super) async fn session(name: &str) -> (Runtime, SessionHandle, Steps) {
    let (builder, steps) = builder(name);
    let mut executor = Executor::new(builder.time.clone(), builder.spawner.clone());
    let (cores, handle) = builder.start(&mut executor).await.unwrap();
    (executor.into_runtime(cores), handle, steps)
}

#[tokio::test(start_paused = true)]
async fn driver_open_defers_host_delivery_until_runtime_executes_effects() {
    let (mut runtime, mut handle, steps) = session("effect-boundary").await;
    while handle.events.try_recv().is_ok() {}
    let before = steps.lock().clone();
    let effects = runtime.driver.open(runtime.shared.time.read());
    assert!(handle.events.try_recv().is_err(), "the decision step delivered a host event");
    assert_eq!(steps.lock().clone(), before, "the decision step invoked the journal observer");
    runtime.perform_voice(effects);
    assert!(handle.events.try_recv().is_ok(), "runtime never delivered the opening event");
    assert!(
        steps.lock().clone().len() > before.len(),
        "runtime never delivered journal observations"
    );
}

#[tokio::test(start_paused = true)]
async fn missing_provider_open_reports_correlated_failure() {
    use lk_agents_voice::agent_session::effects::{Effect, StreamKind};
    let (mut runtime, _, _steps) = session("missing-provider").await;
    while runtime.queue.try_recv().is_ok() {}
    for leg in [StreamKind::Stt, StreamKind::Vad] {
        let gone = lk_agents_abi::resources::ResourceId(u64::MAX);
        runtime.perform_voice(vec![match leg {
            StreamKind::Stt => Effect::SttStreamOpen { open: 42, stt: gone },
            StreamKind::Vad => Effect::VadStreamOpen { open: 42, vad: gone },
        }]);
        assert!(
            matches!(
                runtime.queue.try_recv(),
                Ok(lk_agents_voice::agent_session::Event::LegOpened(
                    lk_agents_voice::room_io::StreamOpened { open: 42, leg: reported, outcome: Err(_) }
                )) if reported == leg
            ),
            "a failed launch must settle its request"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn close_is_serviced_before_ready_media() {
    let (mut runtime, handle, _steps) = session("saturated-media").await;
    let (media, frames) = tokio::sync::mpsc::channel(64);
    for _ in 0..64 {
        media.try_send(lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160)).unwrap();
    }
    runtime.frames = frames;
    handle.api.send(Command::Close(Close::default())).unwrap();
    runtime.run().await;
    assert_eq!(runtime.frames.len(), 64, "ready media delayed close");
}

#[tokio::test(start_paused = true)]
async fn accepted_views_are_published_before_receipts() {
    use lk_agents_voice::agent_session::effects::Effect;
    let (mut runtime, handle, _steps) = session("view-before-receipt").await;
    let view = runtime.shared.view.subscribe();
    let mut accepted = (**view.borrow()).clone();
    accepted
        .config
        .configure(&lk_agents_voice::agent_activity::options::aec_warmup_layer(Some(9.0)));
    // The receipt's waker reads the view at the instant the answer is sent.
    struct ReadsView(
        tokio::sync::watch::Receiver<Arc<lk_agents_voice::agent_session::SessionView>>,
    );
    impl std::task::Wake for ReadsView {
        fn wake(self: Arc<Self>) {
            assert_eq!(
                self.0.borrow().config.effective.timers.aec_warmup(),
                Some(Duration::from_secs(9)),
                "receipt observed options from before acceptance"
            );
        }
    }
    let (call, mut receipt) = runtime.shared.open_call(Answer::interrupt);
    let call_id = call.id;
    let waker = std::task::Waker::from(Arc::new(ReadsView(view.clone())));
    let polled = std::pin::Pin::new(&mut receipt).poll(&mut std::task::Context::from_waker(&waker));
    assert!(polled.is_pending());
    let call = runtime.calls.register(call);
    runtime.handle(
        lk_agents_voice::agent_session::Event::Awaited(
            lk_agents_voice::agent_session::Input::Host(lk_agents_voice::agent_session::Sent {
                origin: Origin::Host,
                command: Command::SetStaticKeyterms(Vec::new()),
            }),
            call,
        ),
        handle.peer.time.read(),
    );
    runtime.perform_voice(vec![
        Effect::Answer(call_id, Answer::Interrupt(Ok(()))),
        Effect::SessionView(Arc::new(accepted)),
    ]);
    assert!(receipt.await.is_ok(), "the receipt was not delivered");
}

struct StartupStt(
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<tokio::sync::Notify>,
    Arc<Mutex<Option<lk_agents_abi::Ts>>>,
);
struct StartupSource(
    Arc<std::sync::atomic::AtomicUsize>,
    Arc<tokio::sync::Notify>,
    Arc<Mutex<Option<lk_agents_abi::Ts>>>,
);
struct StartupSink;

impl lk_agents_abi::StreamSink<lk_agents_abi::AudioFrame> for StartupSink {
    fn push(
        &self,
        _: lk_agents_abi::AudioFrame,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl lk_agents_abi::StreamSource<lk_agents_abi::SpeechEvent> for StartupSource {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::SpeechEvent>, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(std::future::pending())
    }
    fn close(
        self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move {
            *self.2.lock() = Some(deadline);
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.1.notify_one();
            lk_agents_abi::Closed::Clean
        })
    }
}

impl lk_agents_abi::Stt for StartupStt {
    fn label(&self) -> String {
        "startup-stt".into()
    }
    fn capabilities(&self) -> lk_agents_abi::SttCapabilities {
        lk_agents_abi::SttCapabilities { streaming: true, ..Default::default() }
    }
    fn recognize(
        &self,
        _: &[lk_agents_abi::AudioFrame],
        _: Option<&lk_agents_abi::LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::SpeechEvent, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(async move { unreachable!("startup uses streaming recognition") })
    }
    fn stream(
        &self,
        _: Option<&lk_agents_abi::LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            Option<(lk_agents_abi::SttSink, lk_agents_abi::SttSource)>,
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        Box::pin(async move {
            Ok(Some((
                Box::new(StartupSink) as lk_agents_abi::SttSink,
                Box::new(StartupSource(self.0.clone(), self.1.clone(), self.2.clone()))
                    as lk_agents_abi::SttSource,
            )))
        })
    }
}

struct StartupVad(bool);
impl lk_agents_abi::Vad for StartupVad {
    fn label(&self) -> String {
        "startup-vad".into()
    }
    fn capabilities(&self) -> lk_agents_abi::VadCapabilities {
        Default::default()
    }
    fn stream(
        &self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            (lk_agents_abi::VadSink, lk_agents_abi::VadSource),
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        Box::pin(async move {
            if self.0 {
                return Err(lk_agents_abi::errors::ApiError::new("VAD startup failed").into());
            }
            std::future::pending().await
        })
    }
}

#[tokio::test(start_paused = true)]
async fn failed_or_cancelled_startup_closes_each_already_opened_provider_stream() {
    for fails in [true, false] {
        let (mut builder, _) = builder("provider-startup-cleanup");
        let closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let settled = Arc::new(tokio::sync::Notify::new());
        builder.stt = Some(Arc::new(StartupStt(closed.clone(), settled.clone(), Arc::default())));
        builder.vad = Some(Arc::new(StartupVad(fails)));
        let mut executor = Executor::new(builder.time.clone(), builder.spawner.clone());
        let mut opening = Box::pin(builder.start(&mut executor));
        if fails {
            assert!(opening.await.is_err());
        } else {
            std::future::poll_fn(|cx| {
                assert!(opening.as_mut().poll(cx).is_pending());
                std::task::Poll::Ready(())
            })
            .await;
            drop(opening);
        }
        tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
            .await
            .expect("startup abandoned its opened STT stream without closing it");
        assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn unadopted_initial_streams_close_and_release_their_registered_sinks() {
    let (builder, _) = builder("unadopted-provider");
    let resources = builder.resources.clone();
    let closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let settled = Arc::new(tokio::sync::Notify::new());
    let deadline = Arc::new(Mutex::new(None));
    let mut executor = Executor::new(builder.time.clone(), builder.spawner.clone());
    let legs = executor
        .open_legs(
            &resources,
            Some(Arc::new(StartupStt(closed.clone(), settled.clone(), deadline.clone()))),
            None,
        )
        .await
        .unwrap();
    let sink = legs.stt.expect("the initial STT sink was registered");
    assert!(resources.audio_sink(sink).is_some());
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 0);
    builder.time.advance(Duration::from_secs(7));
    let expected_deadline = builder.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    drop(executor);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
        .await
        .expect("an executor dropped before adoption left its provider stream open");
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(*deadline.lock(), Some(expected_deadline));
    assert!(resources.audio_sink(sink).is_none(), "the abandoned initial sink remained registered");
}
