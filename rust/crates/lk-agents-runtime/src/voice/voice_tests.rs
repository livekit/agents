use super::*;
use crate::telemetry::tests::{ENTRYPOINT, Records, labels};
use crate::voice::session::{DriverBuilder, Journal, SessionExecutor, SessionHandle, Timebase};
use lk_agents_abi::metrics::base::AgentMetrics;
use lk_agents_abi::telemetry::decl::Value;
use lk_agents_abi::telemetry::record::{JobKey, SpanContext};
use lk_agents_voice::agent_session::*;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::Step;
use parking_lot::Mutex;
use std::future::Future as _;
use std::time::Duration;

type Steps = Arc<Mutex<Vec<Step>>>;

fn builder(name: &str) -> (DriverBuilder, Steps) {
    let time = Timebase::from_virtual(Arc::new(lk_agents_abi::VirtualClock::new()));
    let config = lk_agents_voice::agent_session::machine::activity::DriverConfig::new(
        Default::default(),
        lk_agents_voice::agent_session::options::SessionLegs::none(),
    );
    let steps: Steps = Arc::default();
    let observed = steps.clone();
    let spawner = Arc::new(lk_agents_net::task::TokioSpawner::current());
    let mut builder = DriverBuilder::new("agent".into(), time, spawner, config);
    builder.journal = Journal::observed(move |step| observed.lock().push(step.clone()));
    builder.session_id = Some(name.into());
    (builder, steps)
}

pub(super) async fn session(name: &str) -> (VoiceRuntime, SessionHandle, Steps) {
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
    use lk_agents_voice::agent_session::effects::{StreamKind, VoiceEffect};
    let (mut runtime, _, _steps) = session("missing-provider").await;
    while runtime.queue.try_recv().is_ok() {}
    for leg in [StreamKind::Stt, StreamKind::Vad] {
        let gone = lk_agents_abi::resources::ResourceId(u64::MAX);
        runtime.perform_voice(vec![match leg {
            StreamKind::Stt => VoiceEffect::OpenSttStream { open: 42, stt: gone, span: None },
            StreamKind::Vad => VoiceEffect::OpenVadStream { open: 42, vad: gone },
        }]);
        assert!(
            matches!(
                runtime.queue.try_recv(),
                Ok(crate::voice::VoiceRuntimeEvent::LegOpened(
                    crate::voice::events::StreamOpened { open: 42, leg: reported, outcome: Err(_) }
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
        let frame = lk_agents_abi::AudioFrame::new(vec![0; 320], 16_000, 1, 160);
        media.try_send(lk_agents_abi::voice::io::InputFrame { frame, head: false }).unwrap();
    }
    runtime.frames = frames;
    handle.api.send(Command::Close(Close::default())).unwrap();
    runtime.run().await;
    assert_eq!(runtime.frames.len(), 64, "ready media delayed close");
}

#[tokio::test(start_paused = true)]
async fn accepted_views_are_published_before_receipts() {
    use lk_agents_voice::agent_session::effects::VoiceEffect;
    let (mut runtime, handle, _steps) = session("view-before-receipt").await;
    let view = runtime.shared.view.clone();
    let mut accepted = (**view.borrow()).clone();
    accepted
        .config
        .configure(&lk_agents_voice::agent_session::options::aec_warmup_layer(Some(9.0)));
    // The receipt's waker reads the view at the instant the answer is sent.
    struct ReadsView(
        tokio::sync::watch::Receiver<Arc<lk_agents_voice::agent_session::SessionView>>,
    );
    impl std::task::Wake for ReadsView {
        fn wake(self: Arc<Self>) {
            assert_eq!(
                self.0.borrow().config.effective().timers.aec_warmup(),
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
        crate::voice::VoiceRuntimeEvent::Awaited(
            lk_agents_voice::agent_session::VoiceEvent::Host(
                lk_agents_voice::agent_session::Sent {
                    origin: VoiceCommandOrigin::Host,
                    command: Command::SetStaticKeyterms(Vec::new()),
                },
            ),
            call,
        ),
        handle.peer.time.read(),
    );
    runtime.perform_voice(vec![
        VoiceEffect::Answer(call_id, Answer::Interrupt(Ok(()))),
        VoiceEffect::SessionView(Arc::new(accepted)),
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
    Arc<std::sync::atomic::AtomicBool>,
);
struct StartupSink(Arc<std::sync::atomic::AtomicBool>);

impl lk_agents_abi::StreamInput<lk_agents_abi::AudioFrame> for StartupSink {
    fn push(
        &self,
        _: lk_agents_abi::AudioFrame,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn end_input(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl lk_agents_abi::StreamOutput<lk_agents_abi::SpeechEvent> for StartupSource {
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
            assert!(
                self.3.load(std::sync::atomic::Ordering::SeqCst),
                "provider input was not ended before closing its source"
            );
            *self.2.lock() = Some(deadline);
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.1.notify_one();
            lk_agents_abi::Closed::Clean
        })
    }
}

impl lk_agents_abi::utils::stream::ProviderStream<lk_agents_abi::SpeechEvent> for StartupSource {}

impl lk_agents_abi::Stt for StartupStt {
    fn label(&self) -> String {
        "startup-stt".into()
    }
    fn capabilities(&self) -> lk_agents_abi::SttCapabilities {
        lk_agents_abi::SttCapabilities { streaming: true, ..Default::default() }
    }
    fn recognize(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
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
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Option<&lk_agents_abi::LanguageCode>,
        _: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            Option<lk_agents_abi::DuplexStream<lk_agents_abi::SttSink, lk_agents_abi::SttSource>>,
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        Box::pin(async move {
            let ended = Arc::new(std::sync::atomic::AtomicBool::new(false));
            Ok(Some(lk_agents_abi::DuplexStream::new(
                Box::new(StartupSink(ended.clone())) as lk_agents_abi::SttSink,
                Box::new(StartupSource(self.0.clone(), self.1.clone(), self.2.clone(), ended))
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
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::VadSink, lk_agents_abi::VadSource>,
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
            lk_agents_abi::telemetry::Cx::NONE,
        )
        .await
        .unwrap();
    let sink = legs.stt.expect("the initial STT sink was registered");
    assert!(resources.audio_sink(sink).is_some());
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 0);
    builder.time.advance(Duration::from_secs(7));
    let expected_deadline = builder.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
    // Startup admission can retire the registry before the executor future is dropped.
    resources.clear();
    drop(executor);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
        .await
        .expect("an executor dropped before adoption left its provider stream open");
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(*deadline.lock(), Some(expected_deadline));
    assert!(resources.audio_sink(sink).is_none(), "the abandoned initial sink remained registered");
}

#[tokio::test(start_paused = true)]
async fn a_tap_refused_by_the_closed_queue_is_retired_without_a_runtime_cleanup_pass() {
    let (mut runtime, handle, _) = session("late-audio-tap").await;
    runtime.queue.close();
    let ended = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sink = Arc::new(CountedTap(ended.clone()));
    let retained = Arc::downgrade(&sink);
    assert!(handle.peer.install_audio_tap(sink).is_err());
    assert_eq!(ended.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(retained.upgrade().is_none());
    assert!(runtime.shared.resources.live().is_empty());
}

#[tokio::test(start_paused = true)]
async fn retiring_a_queue_ends_every_tap_in_an_unprocessed_burst() {
    for dropped in [false, true] {
        let (mut runtime, handle, _) = session("queued-tap-burst").await;
        let resources = runtime.shared.resources.clone();
        let mut taps = Vec::new();
        for _ in 0..512 {
            let ended = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let tap = Arc::new(CountedTap(ended.clone()));
            let retained = Arc::downgrade(&tap);
            handle.peer.install_audio_tap(tap).unwrap();
            taps.push((ended, retained));
        }
        if !dropped {
            runtime.close().await;
        }
        drop(runtime);
        assert!(handle.api.is_closed(), "resource cleanup preceded queue admission shutdown");
        assert!(resources.live().is_empty(), "queued registrations outlived their queue owner");
        for (ended, retained) in taps {
            assert_eq!(ended.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert!(retained.upgrade().is_none());
        }
    }
}

struct CountedTap(Arc<std::sync::atomic::AtomicUsize>);
impl lk_agents_abi::StreamInput<lk_agents_abi::AudioFrame> for CountedTap {
    fn push(
        &self,
        _: lk_agents_abi::AudioFrame,
    ) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), lk_agents_abi::errors::ProviderError> {
        Ok(())
    }
    fn end_input(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

async fn prepared_stt(
    runtime: &VoiceRuntime,
) -> (crate::voice::events::StreamOpened, Arc<tokio::sync::Notify>) {
    let settled = Arc::new(tokio::sync::Notify::new());
    let provider = StartupStt(Arc::default(), settled.clone(), Arc::default());
    let stream =
        lk_agents_abi::Stt::stream(&provider, &lk_agents_abi::telemetry::Cx::NONE, None, None)
            .await
            .unwrap()
            .unwrap();
    let (sink, source) = stream.split();
    (
        crate::voice::events::StreamOpened {
            open: 123,
            leg: lk_agents_voice::agent_session::effects::StreamKind::Stt,
            outcome: Ok(crate::voice::events::Opened::new(
                lk_agents_voice::audio::AudioChannel::Stt,
                Arc::from(sink),
                crate::voice::legs::PendingLeg::Stt(source),
                runtime.shared.resources.clone(),
                &runtime.shared.background(),
            )),
        },
        settled,
    )
}

#[tokio::test(start_paused = true)]
async fn rejected_opened_stream_ends_input_before_closing_its_source() {
    let (mut runtime, _, _) = session("refused-opened-stream").await;
    let (opened, settled) = prepared_stt(&runtime).await;
    let VoiceEvent::StreamOpened { outcome: Ok(sink), .. } = runtime.prepare_leg(opened) else {
        panic!("prepared STT did not produce a resource identity");
    };
    runtime.perform_voice(vec![
        lk_agents_voice::agent_session::effects::VoiceEffect::DiscardOpenedInput(sink),
    ]);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
        .await
        .expect("the discarded provider source did not close");
    assert!(runtime.shared.resources.audio_sink(sink).is_none());
    runtime.close().await;
}

#[tokio::test(start_paused = true)]
async fn dropping_runtime_closes_a_provider_result_still_in_its_queue() {
    let (runtime, handle, _) = session("queued-provider-result").await;
    let resources = runtime.shared.resources.clone();
    let (opened, settled) = prepared_stt(&runtime).await;
    assert!(handle.api.send_opened(opened).is_ok());
    drop(runtime);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
        .await
        .expect("dropping the queue abandoned a successful provider open");
    assert!(resources.live().is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_successful_provider_open_refused_by_the_queue_keeps_its_cleanup_owner() {
    let (mut runtime, _, _) = session("refused-provider-completion").await;
    let settled = Arc::new(tokio::sync::Notify::new());
    let closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let provider = StartupStt(closed.clone(), settled.clone(), Arc::default());
    let stt = runtime.shared.resources.register_stt(Arc::new(provider));
    runtime.queue.close();
    runtime.perform_voice(vec![
        lk_agents_voice::agent_session::effects::VoiceEffect::OpenSttStream {
            open: 321,
            stt: stt.id,
            span: None,
        },
    ]);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
        .await
        .expect("queue refusal abandoned a successfully opened stream");
    assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    runtime.close().await;
    assert!(runtime.shared.resources.live().is_empty());
}

#[tokio::test(start_paused = true)]
async fn an_unconsumed_provider_result_owns_its_cleanup() {
    let (runtime, _, _) = session("unconsumed-provider-result").await;
    let (opened, settled) = prepared_stt(&runtime).await;
    assert_eq!(runtime.shared.resources.live().len(), 1);
    drop(opened);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
        .await
        .expect("an unconsumed successful result did not close its provider stream");
    assert!(runtime.shared.resources.live().is_empty());
}

struct CountedInput([std::sync::atomic::AtomicUsize; 2]);
impl lk_agents_abi::voice::io::AudioInput for CountedInput {
    fn label(&self) -> &str {
        "retirement-input"
    }
    fn next_frame(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        lk_agents_abi::voice::io::Result<Option<lk_agents_abi::AudioFrame>>,
    > {
        Box::pin(std::future::pending())
    }
    fn on_attached(&self) {
        self.0[0].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    fn on_detached(&self) {
        self.0[1].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl lk_agents_abi::voice::io::VideoInput for CountedInput {
    fn label(&self) -> &str {
        "retirement-video-input"
    }
    fn next_frame(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        lk_agents_abi::voice::io::Result<Option<lk_agents_abi::voice::rtc::VideoFrameRef>>,
    > {
        Box::pin(std::future::pending())
    }
    fn on_attached(&self) {
        self.0[0].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    fn on_detached(&self) {
        self.0[1].fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[tokio::test(start_paused = true)]
async fn dropping_runtime_detaches_its_live_audio_and_video_sources_once() {
    let (mut runtime, _, _) = session("retiring-inputs").await;
    let input = Arc::new(CountedInput(Default::default()));
    let source = runtime.shared.resources.register_audio_input(input.clone());
    runtime.command(AudioInputRouterCommand::SetInputEnabled(true));
    runtime.command(AudioInputRouterCommand::SetSource(Some(source)));
    let video = Arc::new(CountedInput(Default::default()));
    let camera = runtime.shared.resources.register_video_input(video.clone());
    runtime.perform_voice(vec![lk_agents_voice::agent_session::effects::VoiceEffect::Audio(
        lk_agents_voice::agent_session::effects::AudioOp::SetVideoInput(Some(camera.id)),
    )]);
    for source in [&input, &video] {
        assert_eq!(source.0[0].load(std::sync::atomic::Ordering::SeqCst), 1);
    }
    drop(runtime);
    for (source, label) in [(&video, "video"), (&input, "audio")] {
        assert_eq!(
            source.0[1].load(std::sync::atomic::Ordering::SeqCst),
            1,
            "retiring the queue owner left its {label} source attached"
        );
    }
}

#[tokio::test(start_paused = true)]
async fn refused_cleanup_task_admission_preserves_the_stream_owner() {
    let (mut runtime, _, _) = session("refused-cleanup-task").await;
    let (opened, settled) = prepared_stt(&runtime).await;
    let VoiceEvent::StreamOpened { outcome: Ok(sink), .. } = runtime.prepare_leg(opened) else {
        panic!("prepared STT did not produce a resource identity");
    };
    runtime.shared.tasks.close();
    runtime.perform_voice(vec![
        lk_agents_voice::agent_session::effects::VoiceEffect::DiscardOpenedInput(sink),
    ]);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
        .await
        .expect("task refusal abandoned the armed stream owner");
    assert!(runtime.shared.resources.live().is_empty());
}

#[test]
fn watch_close_preserves_queued_order_and_refuses_late_subscriptions() {
    let watch = crate::voice::session::WatchHub::new();
    let mut feed = watch.subscribe();
    watch.publish(SystemTap::Ivr(IvrCommand::Close));
    watch.publish(SystemTap::CloseRequested);
    watch.close();
    watch.publish(SystemTap::CloseRequested);
    assert!(matches!(feed.try_recv(), Ok(SystemTap::Ivr(IvrCommand::Close))));
    assert!(matches!(feed.try_recv(), Ok(SystemTap::CloseRequested)));
    assert!(matches!(feed.try_recv(), Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)));
    assert!(watch.subscribe().is_closed());
    assert!(watch.is_empty());
}

#[derive(Default)]
struct RealtimeProbe {
    closing: tokio::sync::Notify,
    release: Option<Arc<tokio::sync::Notify>>,
    closed: std::sync::atomic::AtomicUsize,
    reading: Arc<std::sync::atomic::AtomicBool>,
    settled: tokio::sync::Notify,
}

impl lk_agents_abi::RealtimeSession for RealtimeProbe {
    fn capabilities(&self) -> lk_agents_abi::llm::realtime::RealtimeCapabilities {
        Default::default()
    }
    fn send(&self, _: lk_agents_abi::llm::realtime::RealtimeCommand) {}
    fn head(&self, _: lk_agents_abi::AudioFrame) {}
    fn next_event(
        &self,
    ) -> lk_agents_abi::BoxFuture<'_, Option<lk_agents_abi::llm::realtime::Event>> {
        struct Reading(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Reading {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }
        Box::pin(async move {
            self.reading.store(true, std::sync::atomic::Ordering::SeqCst);
            let _reading = Reading(self.reading.clone());
            std::future::pending().await
        })
    }
    fn chat_ctx(&self) -> lk_agents_abi::BoxFuture<'_, lk_agents_abi::ChatContext> {
        Box::pin(async { Default::default() })
    }
    fn aclose(&self) -> lk_agents_abi::BoxFuture<'_, ()> {
        Box::pin(async move {
            self.closed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.closing.notify_one();
            if let Some(release) = &self.release {
                release.notified().await;
            }
            self.settled.notify_one();
        })
    }
}

#[tokio::test(start_paused = true)]
async fn abandoned_realtime_connection_and_its_pending_pump_close() {
    let (mut runtime, _, _) = session("abandoned-realtime").await;
    let probe = Arc::new(RealtimeProbe::default());
    runtime.realtime =
        Some(legs::RealtimeConnection::new(probe.clone(), &runtime.shared.background()));
    runtime.pump_realtime(probe.clone());
    tokio::task::yield_now().await;
    assert!(probe.reading.load(std::sync::atomic::Ordering::SeqCst));
    drop(runtime);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, probe.settled.notified())
        .await
        .expect("realtime connection was abandoned without closing");
    tokio::task::yield_now().await;
    assert!(!probe.reading.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(probe.closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

/// The builder's executor and clock, over `tasks`.
fn on(
    builder: &DriverBuilder,
    tasks: &Arc<lk_agents_net::task::TaskSet>,
) -> lk_agents_net::owned::Background {
    lk_agents_net::owned::Background::new(
        Arc::clone(tasks),
        builder.spawner.clone(),
        builder.time.clock(),
    )
}

#[tokio::test(start_paused = true)]
async fn realtime_close_is_owned_once_even_when_task_admission_is_refused() {
    let (builder, _) = builder("refused-realtime-close");
    let probe = Arc::new(RealtimeProbe::default());
    let tasks = Arc::<lk_agents_net::task::TaskSet>::default();
    tasks.close();
    let mut connection = legs::RealtimeConnection::new(probe.clone(), &on(&builder, &tasks));
    let tx = crate::voice::session::Tx::disconnected();
    connection.close(tx.clone());
    connection.close(tx);
    drop(connection);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, probe.settled.notified())
        .await
        .expect("realtime connection was abandoned without closing");
    assert_eq!(probe.closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

struct RealtimeModelProbe(Arc<RealtimeProbe>);
impl lk_agents_abi::Realtime for RealtimeModelProbe {
    fn label(&self) -> String {
        "owned-realtime".into()
    }
    fn capabilities(&self) -> lk_agents_abi::llm::realtime::RealtimeCapabilities {
        Default::default()
    }
    fn session(
        &self,
        _: lk_agents_abi::telemetry::Cx,
        _: bool,
        _: Option<lk_agents_abi::HostRef>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Arc<dyn lk_agents_abi::RealtimeSession>, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(async { Ok(self.0.clone() as Arc<dyn lk_agents_abi::RealtimeSession>) })
    }
}

#[tokio::test(start_paused = true)]
async fn realtime_open_is_owned_before_cores_or_runtime_exist() {
    let (builder, _) = builder("realtime-startup-owner");
    let probe = Arc::new(RealtimeProbe::default());
    let mut executor = Executor::new(builder.time.clone(), builder.spawner.clone());
    let _ = executor
        .open_realtime(
            &builder.resources,
            Arc::new(RealtimeModelProbe(probe.clone())),
            false,
            None,
            lk_agents_abi::telemetry::Cx::NONE,
        )
        .await
        .unwrap();
    drop(executor);
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, probe.settled.notified())
        .await
        .expect("realtime connection was abandoned without closing");
    assert_eq!(probe.closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn cancelling_a_polled_realtime_close_continues_the_same_operation() {
    let (builder, _) = builder("cancel-polled-realtime-close");
    let release = Arc::new(tokio::sync::Notify::new());
    let probe = Arc::new(RealtimeProbe { release: Some(release.clone()), ..Default::default() });
    let tasks = Arc::<lk_agents_net::task::TaskSet>::default();
    let mut connection = legs::RealtimeConnection::new(probe.clone(), &on(&builder, &tasks));
    connection.close(crate::voice::session::Tx::disconnected());
    probe.closing.notified().await;
    tasks.close();
    tokio::task::yield_now().await;
    release.notify_one();
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, probe.settled.notified())
        .await
        .expect("cancelling its waiter abandoned a pending realtime close");
    drop(connection);
    assert_eq!(probe.closed.load(std::sync::atomic::Ordering::SeqCst), 1);
}

struct PendingCloseSource {
    probe: Arc<RealtimeProbe>,
    ended: Arc<std::sync::atomic::AtomicBool>,
}
impl lk_agents_abi::StreamOutput<lk_agents_abi::SpeechEvent> for PendingCloseSource {
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
        _: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move {
            assert!(self.ended.load(std::sync::atomic::Ordering::SeqCst));
            lk_agents_abi::RealtimeSession::aclose(self.probe.as_ref()).await;
            lk_agents_abi::Closed::Clean
        })
    }
}
impl lk_agents_abi::utils::stream::ProviderStream<lk_agents_abi::SpeechEvent>
    for PendingCloseSource
{
}

#[tokio::test(start_paused = true)]
async fn cancelling_a_polled_stream_retirement_preserves_close_and_registration_release() {
    let (builder, _) = builder("cancel-polled-stream-close");
    let release = Arc::new(tokio::sync::Notify::new());
    let probe = Arc::new(RealtimeProbe { release: Some(release.clone()), ..Default::default() });
    let ended = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let tasks = Arc::<lk_agents_net::task::TaskSet>::default();
    let opened = legs::OpeningLeg::new(
        lk_agents_voice::audio::AudioChannel::Amd,
        Arc::new(StartupSink(ended.clone())),
        legs::PendingLeg::Stt(Box::new(PendingCloseSource { probe: probe.clone(), ended })),
        builder.resources.clone(),
        &on(&builder, &tasks),
    );
    let id = opened.resource();
    drop(opened);
    probe.closing.notified().await;
    tasks.close();
    tokio::task::yield_now().await;
    release.notify_one();
    tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, probe.settled.notified())
        .await
        .expect("cancelling its waiter abandoned a pending stream close");
    assert_eq!(probe.closed.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(builder.resources.audio_sink(id).is_none());
}

// VAD uses the same source-close contract, with a distinct event type and opening branch.
impl lk_agents_abi::StreamOutput<lk_agents_abi::VadEvent> for StartupSource {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<lk_agents_abi::VadEvent>, lk_agents_abi::errors::ProviderError>,
    > {
        Box::pin(std::future::pending())
    }
    fn close(
        self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        <Self as lk_agents_abi::StreamOutput<lk_agents_abi::SpeechEvent>>::close(self, deadline)
    }
}
impl lk_agents_abi::utils::stream::ProviderStream<lk_agents_abi::VadEvent> for StartupSource {}

struct SuccessfulVad(StartupStt);
impl lk_agents_abi::Vad for SuccessfulVad {
    fn label(&self) -> String {
        "successful-vad".into()
    }
    fn capabilities(&self) -> lk_agents_abi::VadCapabilities {
        Default::default()
    }
    fn stream(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            lk_agents_abi::DuplexStream<lk_agents_abi::VadSink, lk_agents_abi::VadSource>,
            lk_agents_abi::errors::ProviderError,
        >,
    > {
        Box::pin(async move {
            let ended = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let StartupStt(closed, settled, deadline) = &self.0;
            Ok(lk_agents_abi::DuplexStream::new(
                Box::new(StartupSink(ended.clone())) as lk_agents_abi::VadSink,
                Box::new(StartupSource(closed.clone(), settled.clone(), deadline.clone(), ended))
                    as lk_agents_abi::VadSource,
            ))
        })
    }
}

#[tokio::test(start_paused = true)]
async fn successful_vad_results_close_when_refused_or_unread() {
    for refused in [false, true] {
        let (mut runtime, _, _) = session("successful-vad-result").await;
        while runtime.queue.try_recv().is_ok() {}
        let settled = Arc::new(tokio::sync::Notify::new());
        let closed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let deadline = Arc::new(Mutex::new(None));
        let vad = runtime.shared.resources.register_vad(Arc::new(SuccessfulVad(StartupStt(
            closed.clone(),
            settled.clone(),
            deadline.clone(),
        ))));
        let expected_deadline =
            runtime.shared.time.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
        if refused {
            runtime.queue.close();
        }
        runtime.perform_voice(vec![
            lk_agents_voice::agent_session::effects::VoiceEffect::OpenVadStream {
                open: 654,
                vad: vad.id,
            },
        ]);
        if !refused {
            let result = tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, async {
                while let Some(event) = runtime.queue.recv().await {
                    if let VoiceRuntimeEvent::LegOpened(opened) = event {
                        return opened;
                    }
                }
                panic!("queue ended before its successful VAD result");
            })
            .await
            .expect("VAD open did not return a result");
            assert!(matches!(
                &result,
                crate::voice::events::StreamOpened {
                    open: 654,
                    leg: lk_agents_voice::agent_session::effects::StreamKind::Vad,
                    outcome: Ok(_),
                }
            ));
            drop(result);
        }
        tokio::time::timeout(lk_agents_abi::HOST_TEARDOWN_TIMEOUT, settled.notified())
            .await
            .expect("successful VAD result lost its cleanup owner");
        assert_eq!(closed.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(*deadline.lock(), Some(expected_deadline));
        assert_eq!(runtime.shared.resources.live().len(), 1, "VAD sink remained registered");
        runtime.close().await;
        assert!(runtime.shared.resources.live().is_empty());
    }
}

const JOB: &str = "AJ_1";

pub(crate) async fn traced(name: &str) -> (VoiceRuntime, Arc<Records>) {
    let records = Arc::new(Records::default());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let time = Timebase::from_virtual(clock.clone());
    let config = lk_agents_voice::agent_session::machine::activity::DriverConfig::new(
        Default::default(),
        lk_agents_voice::agent_session::options::SessionLegs::none(),
    );
    let spawner = Arc::new(lk_agents_net::task::TokioSpawner::current());
    let mut builder = DriverBuilder::new("agent".into(), time, spawner, config);
    builder.session_id = Some(name.into());
    builder.tracer = lk_agents_telemetry::Tracer::new(records.clone(), clock, 5);
    builder.trace =
        builder.tracer.root().continuing(Some(ENTRYPOINT)).with_job(Some(JobKey(JOB.into())));
    let mut executor = Executor::new(builder.time.clone(), builder.spawner.clone());
    let (cores, _handle) = builder.start(&mut executor).await.unwrap();
    (executor.into_runtime(cores), records)
}

// Catches: the session's spans leaving the job's trace or its Cloud scope.
#[tokio::test(start_paused = true)]
async fn the_session_joins_the_jobs_trace() {
    let (runtime, _records) = traced("joined").await;
    let start = runtime.driver.session_start_span().expect("session_start is open");
    assert_eq!(start.trace_id, 3, "the session continues the entrypoint's trace");
    assert_eq!(runtime.shared.trace.job(), Some(&JobKey(JOB.into())));
}

// Catches: a session dropped without its close (a cut-off loop) losing its spans, the
// `agent_session` the dashboard hangs every row on among them.
#[tokio::test(start_paused = true)]
async fn a_session_dropped_without_its_close_still_exports_its_spans() {
    let (runtime, records) = traced("abandoned").await;
    assert!(records.spans().iter().all(|s| s.name != "agent_session"), "still open");
    drop(runtime);
    let spans = records.spans();
    let names: Vec<_> = spans.iter().map(|s| s.name.as_ref()).collect();
    assert!(names.contains(&"agent_session"), "{names:?}");
    assert!(names.contains(&"session_start"), "{names:?}");
}

// Catches: usage counted under GenAI labels instead of the model labels, a zero counted, or
// the connection time missing its reuse label.
#[tokio::test(start_paused = true)]
async fn usage_counters_carry_the_model_labels() {
    let (mut runtime, records) = traced("usage").await;
    let mut tts = lk_agents_abi::metrics::testkit::tts();
    tts.metadata = lk_agents_abi::metrics::testkit::meta("cartesia", "sonic");
    tts.characters_count = 12;
    tts.audio_duration = 0.0;
    tts.acquire_time = Some(0.2);
    tts.connection_reused = true;
    runtime.deliver_session_event(SessionEvent::MetricsCollected {
        metrics: Box::new(AgentMetrics::Tts(tts)),
        speech: None,
        counted: false,
    });
    let characters = records.measurements("lk.agents.usage.tts_characters");
    assert_eq!(characters.len(), 1);
    assert_eq!(
        labels(&characters[0]),
        [
            ("model_provider".to_owned(), Value::String("cartesia".into())),
            ("model_name".to_owned(), Value::String("sonic".into())),
        ]
    );
    assert_eq!(characters[0].job, Some(JobKey(JOB.into())));
    assert!(records.measurements("lk.agents.usage.tts_audio_duration").is_empty());
    let acquire = records.measurements("lk.agents.connection.acquire_time");
    assert!(
        labels(&acquire[0]).contains(&("connection_reused".into(), Value::String("true".into())))
    );
}

/// A job whose room connection records the span it ran under.
#[derive(Default)]
struct Job(Mutex<Vec<Option<SpanContext>>>);

impl lk_agents_abi::worker::job::SessionJob for Job {
    fn connect(
        &self,
        cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        self.0.lock().push(cx.span());
        Box::pin(async { Ok(()) })
    }
    fn enable_recording(&self) -> Result<bool, lk_agents_abi::errors::HostError> {
        unreachable!("only the session owner joins its job")
    }
    fn claim_primary(
        &self,
        _: lk_agents_abi::HostRef,
    ) -> Result<bool, lk_agents_abi::errors::HostError> {
        unreachable!("only the session owner joins its job")
    }
    fn recording_dir(&self) -> Result<String, lk_agents_abi::errors::HostError> {
        unreachable!("only the session owner joins its job")
    }
    fn init_recording(
        &self,
        _: std::collections::BTreeMap<String, bool>,
    ) -> Result<(), lk_agents_abi::errors::HostError> {
        unreachable!("only the session owner joins its job")
    }
    fn run(&self) -> Arc<dyn lk_agents_abi::worker::ipc_host::JobExecution> {
        unreachable!("only the session owner joins its job")
    }
    fn server_env(
        &self,
    ) -> Result<std::collections::BTreeMap<String, String>, lk_agents_abi::errors::HostError> {
        unreachable!("only an avatar calls the job's RoomService")
    }
}

// Catches: the room connecting before `session_start` exists, so `room_connect` parents to the
// job instead of the session's start; the driver drawing spans other than those `start()`
// reserved, which the startup work already ran under; or a session span recording its own session
// id instead of the call's conversation (its thread).
#[tokio::test(start_paused = true)]
async fn startup_runs_under_the_spans_start_reserved() {
    use lk_agents_abi::telemetry::spans::session::{AgentSession, SessionStart};
    let records = Arc::new(Records::default());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let tracer = lk_agents_telemetry::Tracer::new(records.clone(), clock.clone(), 5);
    let env = lk_agents_net::env::ProcessEnv {
        tracer: tracer.clone(),
        ..lk_agents_net::env::ProcessEnv::current_runtime()
    };
    let time = Timebase::from_virtual(clock);
    let room = crate::room_io::FixedRoom { rtc: None, time, pool: None };
    let job = Arc::new(Job::default());
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "agent".into(),
        job: Some(job.clone()),
        ..Default::default()
    };
    let session = <VoiceSession as lk_agents_abi::voice::session::VoiceSession>::new(
        &env,
        Arc::new(room),
        args,
        None,
    )
    .unwrap();
    let trace = (tracer.root().continuing(Some(ENTRYPOINT)))
        .with_job(Some(JobKey(JOB.into())))
        .with_conversation(Some("AT_thread"));
    let reserved = crate::telemetry::reserve::<AgentSession>(&trace);
    let start = crate::telemetry::reserve::<SessionStart>(
        &trace.clone().under::<AgentSession>(reserved.context),
    );
    let opening = lk_agents_voice::telemetry::Opening { session: reserved, start, activity: None };
    session.trace_under(trace, Default::default(), opening);
    use lk_agents_abi::voice::session::VoiceSession as _;
    session.start().unwrap().await.unwrap();
    assert_eq!(*job.0.lock(), [Some(start.context)], "the room connects under session_start");
    session.close_session(lk_agents_abi::voice::events::CloseReason::UserInitiated, false).await;
    let spans = records.spans();
    let named = |name: &str| spans.iter().find(|s| s.name == name).expect(name).clone();
    let root = named("agent_session");
    assert_eq!((root.context, root.parent), (reserved.context, Some(ENTRYPOINT.span_id)));
    assert_eq!(root.start_unix_nanos, reserved.start, "the session starts at start()");
    let started = named("session_start");
    assert_eq!((started.context, started.parent), (start.context, Some(root.context.span_id)));
    let key = lk_agents_abi::telemetry::semconv::otel::GEN_AI_CONVERSATION_ID;
    for span in &spans {
        let conversation = span.attributes.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        assert_eq!(
            conversation,
            Some(&lk_agents_abi::telemetry::record::Value::String("AT_thread".into())),
            "{}",
            span.name
        );
    }
}
