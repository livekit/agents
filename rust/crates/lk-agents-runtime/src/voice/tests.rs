//! Voice execution contracts: core effects, correlated failures, queue priority and accepted views.

use super::*;
use lk_agents_voice::agent_session::*;
use lk_agents_voice::trace::{Journal, Step};
use std::future::Future as _;
use std::sync::Mutex;
use std::time::Duration;

type Steps = Arc<Mutex<Vec<Step>>>;

async fn session(name: &str) -> (Runtime, SessionHandle, Steps) {
    let time = Timebase::from_virtual(Arc::new(lk_agents_abi::VirtualClock::new()));
    let config = lk_agents_voice::agent_activity::DriverConfig::new(
        Default::default(),
        lk_agents_voice::agent_activity::options::SessionLegs::none(),
    );
    let (view, _) = tokio::sync::watch::channel(Arc::new(SessionView {
        config: config.clone(),
        agent: "agent".into(),
        vad: None,
    }));
    let steps: Steps = Arc::default();
    let observed = steps.clone();
    let builder = DriverBuilder {
        view,
        time,
        config,
        spawner: Arc::new(lk_agents_abi::TokioSpawner::current()),
        journal: Journal::observed(move |step| observed.lock().unwrap().push(step.clone())),
        probe: None,
        agent: "agent".into(),
        session_id: name.into(),
        resources: Default::default(),
        instructions: Default::default(),
        dynamic_instructions: None,
        output_nodes: Default::default(),
        chat_ctx: None,
        hook_bound: None,
        capture_run: None,
        max_unrecoverable_errors: 3,
        conn: Default::default(),
        expressive: Default::default(),
        llm: None,
        tts: None,
        tts_node: None,
        stt_node: None,
        stt: None,
        keyterms: None,
        forward_chat_context: true,
        vad: None,
        audio_in: None,
        video_in: None,
        storage: None,
        recorder: None,
        audio_out: None,
        text_out: None,
        room_events: None,
        room_io_events: None,
        room_link: None,
        eot: None,
        interruption: None,
        video_sampler: None,
        realtime: None,
        toolset: Default::default(),
        // `dst` is enabled by the runtime's test dependency so SessionCores has the same shape.
    };
    let mut executor = Executor::default();
    let (cores, handle) = builder.start(&mut executor).await.unwrap();
    (Runtime::new(cores, Vec::new()), handle, steps)
}

#[tokio::test(start_paused = true)]
async fn driver_open_defers_host_delivery_until_runtime_executes_effects() {
    let (mut runtime, mut handle, steps) = session("effect-boundary").await;
    while handle.events.try_recv().is_ok() {}
    let before = steps.lock().unwrap().clone();
    let effects = runtime.driver.open(runtime.shared.time.read());
    assert!(handle.events.try_recv().is_err(), "the decision step delivered a host event");
    assert_eq!(
        steps.lock().unwrap().clone(),
        before,
        "the decision step invoked the journal observer"
    );
    runtime.perform_voice(effects);
    assert!(handle.events.try_recv().is_ok(), "runtime never delivered the opening event");
    assert!(
        steps.lock().unwrap().clone().len() > before.len(),
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
