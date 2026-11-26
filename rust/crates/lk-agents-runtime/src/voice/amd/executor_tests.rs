//! A run's lifecycle on a real session with no room: entry checks, completion, `execute()` and
//! the models AMD closes. Turn-level behavior over a call is in the DST
//! (`lk-agents-fuzz/tests/regressions/amd/executor.rs`). Expected values come from
//! `voice/amd/detector.py` and `tests/test_amd.py` on `main`.

use super::*;
use crate::room_io::FixedRoom;
use crate::voice::Executor;
use crate::voice::session::{CloseOptions, SessionRunner, Timebase};
use lk_agents_abi::utils::stream::{Closed, ProviderStream, StreamInput, StreamOutput};
use lk_agents_abi::voice::amd as abi;
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::voice::rtc::{
    ByteStreamIngress, ConnectionState, LocalParticipant, ParticipantIdentity, RemoteParticipant,
    RoomEventSink, RtcError, TextStreamIngress,
};
use lk_agents_abi::worker::remote_session::SessionHosting;
use lk_agents_abi::{AudioFrame, ChatRequest, LlmStream, SpeechEvent, Ts};
use lk_agents_voice::agent_session::Command;
use lk_agents_voice::agent_session::VoiceHostEvent;
use lk_agents_voice::agent_session::commands::HookOutcome;
use lk_agents_voice::amd::AmdObservation;
use lk_agents_voice::amd::events::Category;
use lk_agents_voice::amd::events::{AmdReason, Stage};
use lk_agents_voice::audio_recognition::TurnId;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// A session's room without RTC, on Tokio's paused clock: AMD listens at once (`detector.py`'s
/// `_setup_listening` starts listening when the session has no room I/O).
fn no_room() -> FixedRoom {
    let time = Timebase::of_clock(Arc::new(lk_agents_net::clock::RealClock::new()));
    FixedRoom::new(None, time, None)
}

/// A room-only session, started, over a host that answers every hook at once.
async fn session() -> VoiceSession {
    session_with(None).await
}

/// [`session`], serving `remote` as its remote session host.
async fn session_with(remote: Option<Arc<Remote>>) -> VoiceSession {
    let room = Arc::new(no_room()) as Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>;
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "agent".into(),
        room_io_config: Default::default(),
        ..Default::default()
    };
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let api =
        <VoiceSession as lk_agents_abi::voice::session::VoiceSession>::new(&env, room, args, None)
            .expect("valid native room ports");
    let opening = api
        .open(
            Box::new(Executor::new(api.time().clone(), api.spawner())),
            remote.map(|remote| remote as Arc<dyn SessionHosting>),
            move |mut runner: SessionRunner| {
                Box::pin(async move {
                    while let Some(ev) = runner.next_host_event().await {
                        if let VoiceHostEvent::Hook(call) = ev {
                            let _ = runner.api().send(Command::HookDone {
                                id: call.id(),
                                outcome: HookOutcome::Returned,
                            });
                        }
                    }
                })
            },
        )
        .expect("a ready session opens");
    opening.await.expect("the loop starts");
    api
}

/// A model that is never asked anything in these tests.
struct Silent;

impl Llm for Silent {
    fn label(&self) -> String {
        "silent.LLM".into()
    }

    fn chat<'a>(
        &'a self,
        _cx: &Cx,
        _request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<LlmStream, ProviderError>> {
        Box::pin(std::future::pending())
    }
}

/// A streaming STT whose stream hears nothing; `streaming: false` makes it a batch STT.
struct QuietStt {
    streaming: bool,
}

struct Nothing;

impl StreamInput<AudioFrame> for Nothing {
    fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
        Ok(())
    }
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }
    fn end_input(&self) {}
}

impl StreamOutput<SpeechEvent> for Nothing {
    fn next(&mut self) -> BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(std::future::pending())
    }
    fn close(self: Box<Self>, _: Ts) -> BoxFuture<'static, Closed> {
        Box::pin(async { Closed::Clean })
    }
}

impl ProviderStream<SpeechEvent> for Nothing {}

impl Stt for QuietStt {
    fn label(&self) -> String {
        "quiet.STT".into()
    }
    fn capabilities(&self) -> lk_agents_abi::SttCapabilities {
        lk_agents_abi::SttCapabilities { streaming: self.streaming, ..Default::default() }
    }
    fn recognize(
        &self,
        _cx: &Cx,
        _: &[AudioFrame],
        _: Option<&lk_agents_abi::LanguageCode>,
    ) -> BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(std::future::pending())
    }
    fn stream(
        &self,
        _cx: &Cx,
        _: Option<&lk_agents_abi::LanguageCode>,
        _: Option<f64>,
    ) -> BoxFuture<
        '_,
        Result<
            Option<lk_agents_abi::DuplexStream<lk_agents_abi::SttSink, lk_agents_abi::SttSource>>,
            ProviderError,
        >,
    > {
        let streaming = self.streaming;
        Box::pin(async move {
            Ok(streaming.then(|| {
                lk_agents_abi::DuplexStream::new(
                    Box::new(Nothing) as lk_agents_abi::SttSink,
                    Box::new(Nothing) as lk_agents_abi::SttSource,
                )
            }))
        })
    }
}

/// Counts its closes, and whether the run had already returned its outcome at each.
#[derive(Default)]
struct Closes {
    count: AtomicUsize,
    after_outcome: Mutex<Vec<bool>>,
    outcome_seen: std::sync::atomic::AtomicBool,
    /// How long each close takes.
    takes: Duration,
}

impl Closes {
    fn slow(takes: Duration) -> Arc<Self> {
        Arc::new(Closes { takes, ..Closes::default() })
    }

    fn closer(self: &Arc<Self>) -> ModelCloser {
        let closes = Arc::clone(self);
        Box::new(move || {
            Box::pin(async move {
                tokio::time::sleep(closes.takes).await;
                closes.count.fetch_add(1, Ordering::SeqCst);
                let seen = closes.outcome_seen.load(Ordering::SeqCst);
                closes.after_outcome.lock().push(seen);
            })
        })
    }

    fn count(&self) -> usize {
        self.count.load(Ordering::SeqCst)
    }
}

fn dtmf_tool() -> ToolSpec {
    ToolSpec {
        name: "send_dtmf_events".into(),
        description: "Send a list of DTMF events to the telephony provider.".into(),
        json_schema: serde_json::json!({"type": "object", "properties": {}}),
        flags: Default::default(),
        provider_tool: None,
    }
}

fn config(options: AmdOptions) -> AmdConfig {
    AmdConfig {
        options,
        classifier: Classifier::Llm(AmdModel::supplied(Arc::new(Silent) as Arc<dyn Llm>, false)),
        stt: None,
        participant: None,
        dtmf_tool: dtmf_tool(),
        llm_conn_options: ApiConnectOptions::default(),
        stt_conn_options: ApiConnectOptions::default(),
    }
}

/// Waits for `future`, or fails after a minute of the paused clock.
async fn within<T>(what: &str, future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(60), future)
        .await
        .unwrap_or_else(|_| panic!("{what} never happened"))
}

/// Every event the run has sent so far.
fn drain(events: &mut tokio::sync::mpsc::Receiver<AmdEvent>) -> Vec<AmdEvent> {
    let mut out = Vec::new();
    while let Ok(event) = events.try_recv() {
        out.push(event);
    }
    out
}

fn cancelled() -> Outcome {
    Outcome {
        category: Stage::Uncertain,
        reason: AmdReason::Cancelled,
        voicemail_message_played: false,
        last_prediction: None,
    }
}

// Catches: a close that does not complete the run, an outcome that differs between waiters, or
// `amd_completed` sent twice. Python: `aclose()` finishes with `cancelled`; `execute()` shields one
// completion future (`test_cancelled_execute_waiter_does_not_cancel_detection`).
#[tokio::test(start_paused = true)]
async fn a_close_completes_the_run_once_with_cancelled() {
    let api = session().await;
    let run = enter(&api, config(AmdOptions::default())).await.expect("the run attaches");
    let mut events = run.events().expect("the events are taken once");
    assert!(run.events().is_none(), "the events are taken once");

    // A waiter that stops waiting does not cancel the run.
    let abandoned = run.execute();
    drop(abandoned);
    let waiting = run.execute();
    let closed = within("the close", run.close()).await;
    assert_eq!(closed, cancelled());
    assert_eq!(within("execute()", waiting).await, cancelled());
    assert_eq!(within("a later execute()", run.execute()).await, cancelled());

    let events = drain(&mut events);
    assert_eq!(events, vec![AmdEvent::Completed(cancelled())], "one `amd_completed`, last");
}

// Catches: a run that never starts its overall deadline when the session has no room, or starts
// it before listening. Python: with no room I/O `_start_listening` runs at once, and
// `_hard_deadline = time.monotonic() + timeout`.
#[tokio::test(start_paused = true)]
async fn without_a_room_the_run_listens_at_once_and_its_overall_limit_runs() {
    let api = session().await;
    let options = AmdOptions { timeout: Duration::from_secs(3), ..AmdOptions::default() };
    let entered = tokio::time::Instant::now();
    let run = enter(&api, config(options)).await.expect("the run attaches");
    let outcome = within("the overall timeout", run.execute()).await;
    assert_eq!(outcome.reason, AmdReason::Timeout);
    assert_eq!(outcome.category, Stage::Uncertain);
    assert!(!outcome.is_error(), "a timeout is a normal outcome (spec §12.1)");
    assert_eq!(entered.elapsed(), Duration::from_secs(3), "the limit counts from activation");
}

// Catches: a run that outlives its session. Python:
// `test_session_close_completes_amd_and_releases_resources`.
#[tokio::test(start_paused = true)]
async fn the_session_close_completes_the_run_with_cancelled() {
    let api = session().await;
    let run = enter(&api, config(AmdOptions::default())).await.expect("the run attaches");
    within(
        "the session close",
        api.close(CloseReason::UserInitiated, CloseOptions::default(), None),
    )
    .await;
    assert_eq!(within("execute()", run.execute()).await, cancelled());
}

// Catches: models AMD created left open, closed twice, or closed after `execute()` returned.
// Python: `_cleanup` closes the owned models (`_close_resources`) before it sets the completion:
// `test_default_models_are_auto_selected_and_closed`; spec §14 invariant 11.
#[tokio::test(start_paused = true)]
async fn the_models_amd_created_are_closed_once_before_the_outcome() {
    let api = session().await;
    let (llm_closes, stt_closes) = (Arc::new(Closes::default()), Arc::new(Closes::default()));
    let mut config = config(AmdOptions::default());
    config.classifier = Classifier::Llm(AmdModel {
        model: Arc::new(Silent) as Arc<dyn Llm>,
        inference: true,
        close: Some(llm_closes.closer()),
    });
    config.stt = Some(AmdModel {
        model: Arc::new(QuietStt { streaming: true }) as Arc<dyn Stt>,
        inference: true,
        close: Some(stt_closes.closer()),
    });
    let run = enter(&api, config).await.expect("the run attaches");
    assert_eq!((llm_closes.count(), stt_closes.count()), (0, 0), "open while the run runs");

    let outcome = within("the close", run.close()).await;
    llm_closes.outcome_seen.store(true, Ordering::SeqCst);
    stt_closes.outcome_seen.store(true, Ordering::SeqCst);
    assert_eq!(outcome.reason, AmdReason::Cancelled);
    assert_eq!((llm_closes.count(), stt_closes.count()), (1, 1), "each owned model closed once");
    assert_eq!(*llm_closes.after_outcome.lock(), vec![false], "closed before the outcome");
    assert_eq!(*stt_closes.after_outcome.lock(), vec![false], "closed before the outcome");

    within("a second close", run.close()).await;
    assert_eq!((llm_closes.count(), stt_closes.count()), (1, 1), "never closed again");
}

// Catches: a second run attached beside the first, or a refused second run that cancels the
// first (its route replaced, the first run's executor reading the end of its observations).
// Spec §5.1: one run at a time; Python raises "amd is already active" and leaves the first run
// alone (`detector.py` `__aenter__`).
#[tokio::test(start_paused = true)]
async fn a_second_run_is_refused_while_one_is_attached() {
    let api = session().await;
    let options = AmdOptions { timeout: Duration::from_secs(5), ..AmdOptions::default() };
    let first = enter(&api, config(options.clone())).await.expect("the first run attaches");
    let mut events = first.events().expect("the first run's events");
    let second = enter(&api, config(AmdOptions::default())).await;
    assert!(
        matches!(second, Err(AmdEnterError::Refused(AmdAttachRefused::AlreadyAttached))),
        "{:?}",
        second.err()
    );
    // The refused run changed nothing: the first runs on, and ends on its own deadline.
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_eq!(drain(&mut events), vec![], "the first run is still running");
    let outcome = within("the first run's overall limit", first.execute()).await;
    assert_eq!(outcome.reason, AmdReason::Timeout, "the first run was not cancelled");
    let third = enter(&api, config(AmdOptions::default())).await;
    assert!(third.is_ok(), "a new run attaches once the first detached: {:?}", third.err());
    assert_ne!(third.as_ref().map(AmdRun::id).ok(), Some(first.id()), "a new run id");
}

// Catches: a caller that stops waiting for `enter()` after the session accepted the attach,
// leaving the gate attached with no executor (replies held for good) and the models AMD created
// open. Python cleans up a failed `__aenter__`, cancellation included (`detector.py`: `await
// self.aclose(); raise`).
#[tokio::test(start_paused = true)]
async fn a_caller_that_stops_waiting_leaves_no_run_behind() {
    // The entry is polled once, so the run's task exists; each count of yields then drops it at
    // another point: before the session took the attach, after it accepted, and after the answer
    // reached `enter`, unread. (A future never polled started nothing.)
    for yields in 0..8 {
        let api = session().await;
        let closes = Arc::new(Closes::default());
        let mut config = config(AmdOptions::default());
        config.classifier = Classifier::Llm(AmdModel {
            model: Arc::new(Silent) as Arc<dyn Llm>,
            inference: true,
            close: Some(closes.closer()),
        });
        let mut entering = Box::pin(enter(&api, config));
        let first = std::future::poll_fn(|cx| std::task::Poll::Ready(entering.as_mut().poll(cx)));
        assert!(first.await.is_pending(), "the entry waits for the session's answer");
        for _ in 0..yields {
            tokio::task::yield_now().await;
        }
        drop(entering);
        let closed = async {
            while closes.count() == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        // Promptly, not at the run's own 60 s limit.
        tokio::time::timeout(Duration::from_secs(5), closed)
            .await
            .unwrap_or_else(|_| panic!("after {yields} yields: the abandoned run lingers"));
        // The gate is free: a new run attaches.
        let next = within("a new run", enter(&api, config_plain())).await;
        assert!(next.is_ok(), "after {yields} yields: {:?}", next.err());
        tokio::time::sleep(Duration::from_secs(1)).await;
        assert_eq!(closes.count(), 1, "after {yields} yields: each owned model closes once");
    }
}

fn config_plain() -> AmdConfig {
    config(AmdOptions::default())
}

// Catches: an entry the session's AMD task set refuses after the models were handed over, which
// dropped the models AMD created without closing them. Spec §14 invariant 11.
#[tokio::test(start_paused = true)]
async fn a_refused_run_task_still_closes_the_models_amd_created() {
    let api = session().await;
    let (peer, _) = api.amd_peer().expect("running");
    peer.route.tasks().close();
    let (llm_closes, stt_closes) = (Arc::new(Closes::default()), Arc::new(Closes::default()));
    let mut config = config(AmdOptions::default());
    config.classifier = Classifier::Llm(AmdModel {
        model: Arc::new(Silent) as Arc<dyn Llm>,
        inference: true,
        close: Some(llm_closes.closer()),
    });
    config.stt = Some(AmdModel {
        model: Arc::new(QuietStt { streaming: true }) as Arc<dyn Stt>,
        inference: true,
        close: Some(stt_closes.closer()),
    });
    let refused = enter(&api, config).await;
    assert!(
        matches!(refused, Err(AmdEnterError::Session(SessionRefused::Closed))),
        "{:?}",
        refused.err()
    );
    let deadline = api.time().now().saturating_add(Duration::from_secs(30));
    within("the cleanup", peer.route.join(api.time().clock().as_ref(), deadline)).await;
    assert_eq!((llm_closes.count(), stt_closes.count()), (1, 1), "each owned model closed once");
}

// Catches: a session close that returns while its AMD run is still joining its legs and closing
// its models. Python: `_teardown_activity` awaits `self._amd.aclose()`.
#[tokio::test(start_paused = true)]
async fn the_session_close_waits_for_the_run_to_settle() {
    let api = session().await;
    let closes = Closes::slow(Duration::from_secs(2));
    let mut config = config(AmdOptions::default());
    config.classifier = Classifier::Llm(AmdModel {
        model: Arc::new(Silent) as Arc<dyn Llm>,
        inference: true,
        close: Some(closes.closer()),
    });
    let run = enter(&api, config).await.expect("the run attaches");
    within(
        "the session close",
        api.close(CloseReason::UserInitiated, CloseOptions::default(), None),
    )
    .await;
    assert_eq!(closes.count(), 1, "the model closed before the session's close returned");
    assert_eq!(within("execute()", run.execute()).await, cancelled());
}

// Catches: a model close that never returns holding the run's outcome forever.
#[tokio::test(start_paused = true)]
async fn a_model_close_is_bounded_by_the_teardown_bound() {
    let api = session().await;
    let closes = Closes::slow(Duration::from_secs(3600));
    let mut config = config(AmdOptions::default());
    config.classifier = Classifier::Llm(AmdModel {
        model: Arc::new(Silent) as Arc<dyn Llm>,
        inference: true,
        close: Some(closes.closer()),
    });
    let run = enter(&api, config).await.expect("the run attaches");
    let started = tokio::time::Instant::now();
    assert_eq!(within("the close", run.close()).await, cancelled());
    assert!(started.elapsed() <= lk_agents_abi::HOST_TEARDOWN_TIMEOUT + Duration::from_secs(1));
}

/// A room whose events cannot be read: the callee leg dies on its first read.
struct BrokenRoom;

impl lk_agents_abi::voice::rtc::RtcRoom for BrokenRoom {
    fn events(
        &self,
    ) -> lk_agents_abi::voice::rtc::ports::Receiver<lk_agents_abi::voice::rtc::RoomEvent> {
        panic!("injected callee leg bug")
    }
    fn attachments(&self) -> &Arc<dyn lk_agents_abi::voice::room_owner::RoomAttachmentsPort> {
        unreachable!()
    }
    fn rpc_router(&self) -> &Arc<dyn lk_agents_abi::voice::rtc::ports::RpcRoutes> {
        unreachable!()
    }
    fn name(&self) -> Result<String, RtcError> {
        unreachable!()
    }
    fn connection_state(&self) -> Result<ConnectionState, RtcError> {
        unreachable!()
    }
    fn local(&self) -> &Arc<dyn LocalParticipant> {
        unreachable!()
    }
    fn participant(
        &self,
        _: &ParticipantIdentity,
    ) -> Result<Option<Arc<dyn RemoteParticipant>>, RtcError> {
        unreachable!()
    }
    fn remote_participants(&self) -> Result<Vec<Arc<dyn RemoteParticipant>>, RtcError> {
        unreachable!()
    }
    fn event_bus(&self) -> &Arc<dyn RoomEventSink> {
        unreachable!()
    }
    fn register_byte_stream_handler(
        &self,
        _: &str,
        _: Arc<dyn ByteStreamIngress>,
    ) -> Result<(), RtcError> {
        unreachable!()
    }
    fn unregister_byte_stream_handler(&self, _: &str) -> Result<(), RtcError> {
        unreachable!()
    }
    fn register_text_stream_handler(
        &self,
        _: &str,
        _: Arc<dyn TextStreamIngress>,
    ) -> Result<(), RtcError> {
        unreachable!()
    }
    fn unregister_text_stream_handler(&self, _: &str) -> Result<(), RtcError> {
        unreachable!()
    }
}

// Catches: a leg that dies leaving the run waiting on it for good: a callee leg that panics
// while the run is pending (no overall deadline yet). Python: `_on_task_done` ends the run with
// `internal_error`; runtime.md "Task deaths".
#[tokio::test(start_paused = true)]
async fn a_leg_that_dies_ends_the_run_with_internal_error() {
    let api = session().await;
    let callee = CalleeSetup {
        room: Some(Arc::new(BrokenRoom)),
        participant: Some("caller".into()),
        linked: None,
        wait_until_answered: true,
    };
    let run = enter_with(&api, config(AmdOptions::default()), Some(callee))
        .await
        .expect("the run attaches");
    let mut events = run.events().expect("the events");
    let outcome = within("the run's end", run.execute()).await;
    assert_eq!((outcome.category, outcome.reason), (Stage::Uncertain, AmdReason::InternalError));
    assert!(outcome.is_error());
    assert_eq!(drain(&mut events), vec![AmdEvent::Completed(outcome)]);
}

// Catches: options or an STT that `AMD.__init__` rejects accepted at entry. Python:
// `test_max_inference_timeouts_must_be_positive`, `test_nonstreaming_stt_is_rejected`.
#[tokio::test(start_paused = true)]
async fn invalid_options_and_a_batch_stt_are_refused_at_entry() {
    let api = session().await;
    let zero = AmdOptions { inference_timeout: Duration::ZERO, ..AmdOptions::default() };
    let refused = enter(&api, config(zero)).await;
    assert!(
        matches!(refused, Err(AmdEnterError::Options(AmdOptionsError::NotPositive))),
        "{:?}",
        refused.err()
    );

    let mut batch = config(AmdOptions::default());
    batch.stt =
        Some(AmdModel::supplied(Arc::new(QuietStt { streaming: false }) as Arc<dyn Stt>, false));
    let refused = enter(&api, batch).await;
    assert!(matches!(refused, Err(AmdEnterError::NotStreaming)), "{:?}", refused.err());

    // Neither refusal attached a run.
    let run = enter(&api, config(AmdOptions::default())).await;
    assert!(run.is_ok(), "{:?}", run.err());
}

/// A remote session host that records each AMD event it is sent.
#[derive(Default)]
struct Remote(Mutex<Vec<abi::AmdEvent>>);

impl SessionHosting for Remote {
    fn forward(&self, _: &lk_agents_abi::voice::events::HostWire) {}
    fn forward_usage(&self, _: &lk_agents_abi::usage::Usage) {}
    fn forward_error(&self, _: String) {}
    fn forward_tools_executed(&self, _: &lk_agents_abi::voice::events::ToolsExecutedEvent) {}
    fn forward_amd(&self, event: &abi::AmdEvent) {
        self.0.lock().push(event.clone());
    }
    fn start(&self) -> BoxFuture<'_, Result<(), lk_agents_abi::errors::HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn aclose(&self) -> BoxFuture<'_, ()> {
        Box::pin(async {})
    }
}

/// A decision model that chooses `category` for every turn.
struct Chooses(Category);

impl DecisionModel for Chooses {
    fn label(&self) -> String {
        "chooses".into()
    }

    fn choose<'a>(
        &'a self,
        _: &Cx,
        _: &'a ChoiceRequest,
    ) -> BoxFuture<'a, Result<Option<String>, ProviderError>> {
        Box::pin(async move { Ok(Some(self.0.as_str().to_owned())) })
    }
}

// Catches: a prediction or the completion reaching the run's host before the remote session
// host. Spec §13: the remote host gets each prediction and the result first, so a console
// client never sees `amd_completed` after the application reacted to it.
#[tokio::test(start_paused = true)]
async fn the_remote_host_hears_each_event_before_the_runs_host() {
    let remote = Arc::new(Remote::default());
    let api = session_with(Some(Arc::clone(&remote))).await;
    let mut config = config(AmdOptions::default());
    config.classifier = Classifier::Decision(AmdModel::supplied(
        Arc::new(Chooses(Category::Human)) as Arc<dyn DecisionModel>,
        false,
    ));
    let run = enter(&api, config).await.expect("the run attaches");
    let (peer, agent) = api.amd_peer().expect("running");
    let committed = AmdObservation::TurnCommitted {
        turn: TurnId(1),
        transcript: "Hello, who is this?".into(),
        eot_delay: None,
        skip_reply: false,
        bound: true,
        speaking: false,
        agent,
        at: api.time().now(),
    };
    peer.route.deliver(run.id(), &committed);
    let mut heard = Vec::new();
    while let Some(event) = within("the run's next event", run.next_event()).await {
        let event = host::event(&event);
        // The event is already forwarded when the run's host reads it.
        assert_eq!(remote.0.lock().get(heard.len()), Some(&event), "forwarded first");
        heard.push(event);
    }
    assert!(
        matches!(heard.as_slice(), [abi::AmdEvent::Prediction(_), abi::AmdEvent::Completed(_)]),
        "{heard:?}"
    );
    assert_eq!(*remote.0.lock(), heard, "the remote host heard the same events, once each");
}

// Catches: a failing host listener that does not end the run, or ends it with another reason.
// Python: a prediction or menu listener that raises ends the run with `internal_error`
// (`test_menu_listener_error_ends_detection_with_internal_error`); `AmdRun::fail` is that path.
#[tokio::test(start_paused = true)]
async fn a_failed_host_listener_ends_the_run_with_internal_error() {
    let api = session().await;
    let run = enter(&api, config(AmdOptions::default())).await.expect("the run attaches");
    let mut events = run.events().expect("the events");
    run.fail();
    let outcome = within("the run's end", run.execute()).await;
    assert_eq!((outcome.category, outcome.reason), (Stage::Uncertain, AmdReason::InternalError));
    assert_eq!(drain(&mut events), vec![AmdEvent::Completed(outcome.clone())]);
    // Only the first completion counts: a later failure or close changes nothing.
    run.fail();
    assert_eq!(within("a close", run.close()).await, outcome);
}

// Catches: `check` refusing a session that would take an attach, accepting one that would
// refuse it, or attaching anything. Python: `__aenter__`'s session checks run before it creates
// a model (`test_rejected_entry_does_not_create_owned_models`).
#[tokio::test(start_paused = true)]
async fn the_entry_check_answers_as_the_attach_would_and_attaches_nothing() {
    let api = session().await;
    check(&api, false).await.expect("an idle session would take a run");
    check(&api, false).await.expect("a check attaches nothing");
    let run = enter(&api, config(AmdOptions::default())).await.expect("the run attaches");
    let refused = check(&api, false).await;
    assert!(
        matches!(refused, Err(AmdEnterError::Refused(AmdAttachRefused::AlreadyAttached))),
        "{:?}",
        refused.err()
    );
    within("the close", run.close()).await;
    check(&api, false).await.expect("the detached run left the session free");
    within(
        "the session close",
        api.close(CloseReason::UserInitiated, CloseOptions::default(), None),
    )
    .await;
    let closed = check(&api, false).await;
    assert!(matches!(closed, Err(AmdEnterError::Session(_))), "{:?}", closed.err());
}
