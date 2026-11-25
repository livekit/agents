//! A run's lifecycle on a real session with no room: entry checks, completion, `execute()` and
//! the models AMD closes. Turn-level behavior over a call is in the DST
//! (`lk-agents-fuzz/tests/regressions/amd/executor.rs`). Expected values come from
//! `voice/amd/detector.py` and `tests/test_amd.py` on `main`.

use super::*;
use crate::room_io::FixedRoom;
use crate::voice::Executor;
use crate::voice::session::{CloseOptions, SessionRunner, Timebase};
use lk_agents_abi::utils::stream::{Closed, ProviderStream, StreamInput, StreamOutput};
use lk_agents_abi::voice::events::CloseReason;
use lk_agents_abi::{AudioFrame, ChatRequest, LlmStream, SpeechEvent, Ts};
use lk_agents_voice::agent_session::Command;
use lk_agents_voice::agent_session::VoiceHostEvent;
use lk_agents_voice::agent_session::commands::HookOutcome;
use lk_agents_voice::amd::events::{AmdReason, Stage};
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
            None,
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
}

impl Closes {
    fn closer(self: &Arc<Self>) -> ModelCloser {
        let closes = Arc::clone(self);
        Box::new(move || {
            Box::pin(async move {
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
fn drain(events: &mut tokio::sync::mpsc::UnboundedReceiver<AmdEvent>) -> Vec<AmdEvent> {
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

// Catches: a second run attached beside the first. Spec §5.1: one run at a time; Python raises
// "amd is already active".
#[tokio::test(start_paused = true)]
async fn a_second_run_is_refused_while_one_is_attached() {
    let api = session().await;
    let first = enter(&api, config(AmdOptions::default())).await.expect("the first run attaches");
    let second = enter(&api, config(AmdOptions::default())).await;
    assert!(
        matches!(second, Err(AmdEnterError::Refused(AmdAttachRefused::AlreadyAttached))),
        "{:?}",
        second.err()
    );
    // The refused run changed nothing: the first still completes on its own close.
    assert_eq!(within("the first close", first.close()).await, cancelled());
    let third = enter(&api, config(AmdOptions::default())).await;
    assert!(third.is_ok(), "a new run attaches once the first detached: {:?}", third.err());
    assert_ne!(third.as_ref().map(AmdRun::id).ok(), Some(first.id()), "a new run id");
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
