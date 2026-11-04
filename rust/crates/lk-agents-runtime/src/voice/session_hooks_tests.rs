//! The host's answers to the hooks the loop asks for, as the driver receives them.

use super::super::HostAgents;
use super::settle;
use crate::voice::VoiceRuntimeEvent;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::telemetry::record::{SpanId, SpanRef, TraceId};
use lk_agents_abi::voice::session_effects::{FillerReply, SessionHooks};
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use lk_agents_voice::agent_session::{Command, HookCall, HookId, HookOutcome};
use parking_lot::Mutex;
use std::sync::Arc;

struct NoRegistry;

impl lk_agents_abi::RefRegistry for NoRegistry {
    fn release(&self, _id: u64) {}
}

fn host_ref(id: u64) -> lk_agents_abi::HostRef {
    let registry: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRegistry);
    lk_agents_abi::HostRef::new(id, &registry)
}

/// Hooks that answer as configured, and record what failed.
#[derive(Default)]
struct ScriptedHooks {
    raise: bool,
    user_text: Option<String>,
    stop: bool,
    failed: Mutex<Vec<String>>,
    toolsets: ScopeRegistry,
    /// The span each hook ran under.
    under: Mutex<Vec<Option<SpanRef>>>,
}

#[derive(Default)]
struct ScopeRegistry {
    port: std::sync::OnceLock<Arc<dyn lk_agents_abi::llm::toolsets::ScopesPort>>,
    closed: Arc<Mutex<usize>>,
}
struct FakeScopes(Arc<Mutex<usize>>);
impl lk_agents_abi::llm::toolsets::ScopesPort for FakeScopes {
    fn acquire(
        &self,
        _: u64,
        _: Vec<Arc<dyn lk_agents_abi::llm::toolsets::ToolsetHooks>>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<u64, HostError>> {
        Box::pin(async { Ok(1) })
    }
    fn discard(&self, _: u64) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn release(&self, _: Option<u64>) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async {
            *self.0.lock() += 1;
            Ok(())
        })
    }
}

impl ScriptedHooks {
    fn result<T>(&self, value: T) -> Result<T, HostError> {
        if self.raise {
            Err(HostError {
                kind: "ValueError".into(),
                message: "the hook raised".into(),
                host_ref: None,
                traceback: None,
            })
        } else {
            Ok(value)
        }
    }
}

impl SessionHooks for ScriptedHooks {
    fn on_enter<'a>(
        &'a self,
        cx: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        self.under.lock().push(cx.span());
        Box::pin(async move { self.result(()) })
    }
    fn on_exit<'a>(
        &'a self,
        cx: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        self.under.lock().push(cx.span());
        Box::pin(async move { self.result(()) })
    }
    fn toolset_scopes(&self) -> &Arc<dyn lk_agents_abi::llm::toolsets::ScopesPort> {
        self.toolsets.port.get_or_init(|| Arc::new(FakeScopes(self.toolsets.closed.clone())))
    }
    fn toolset_resources(
        &self,
        _: &lk_agents_abi::HostRef,
    ) -> Result<Vec<Arc<dyn lk_agents_abi::llm::toolsets::ToolsetHooks>>, HostError> {
        self.result(Vec::new())
    }
    fn tool_infos(
        &self,
        _: &lk_agents_abi::HostRef,
    ) -> Result<Vec<lk_agents_abi::agent::FunctionToolInfo>, HostError> {
        self.result(Vec::new())
    }
    fn closing(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn on_user_turn_completed<'a>(
        &'a self,
        cx: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::HostRef,
        _: Vec<lk_agents_abi::ChatItem>,
        _: &'a str,
        _: f64,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<(Option<String>, Option<Vec<lk_agents_abi::ChatItem>>, bool), HostError>,
    > {
        self.under.lock().push(cx.span());
        Box::pin(async move { self.result((self.user_text.clone(), Some(Vec::new()), self.stop)) })
    }
    fn on_user_turn_exceeded<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::HostRef,
        _: &'a lk_agents_abi::voice::events::UserTurnExceededEvent,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn function_tools_executed(
        &self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Vec<lk_agents_abi::ChatItem>,
        outputs: Vec<lk_agents_abi::ChatItem>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Vec<bool>, HostError>> {
        Box::pin(async move { self.result(vec![false; outputs.len()]) })
    }
    fn filler_source<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a lk_agents_abi::HostRef,
        step: u32,
    ) -> lk_agents_abi::BoxFuture<'a, Result<FillerReply, HostError>> {
        Box::pin(async move {
            self.result(match step {
                0 => FillerReply::Text("one moment".into()),
                1 => FillerReply::Spoke,
                _ => FillerReply::Declined,
            })
        })
    }
    fn hook_failed(&self, hook: &str, _: Option<lk_agents_abi::errors::CapturedError>) {
        self.failed.lock().push(hook.to_owned());
    }
}

/// What `answer_hook` sends back for `call`.
async fn answered(hooks: &Arc<ScriptedHooks>, call: HookCall) -> HookOutcome {
    use lk_agents_voice::agent_session::{Sent, VoiceEvent};
    let (mut runtime, handle, _) = crate::voice::tests::session("hook-answers").await;
    let agents = HostAgents::default();
    agents.install("agent", 0, Some(host_ref(1)));
    let spawner =
        Arc::new(lk_agents_net::task::TokioSpawner::new(tokio::runtime::Handle::current()));
    let hooks: Arc<dyn SessionHooks> = hooks.clone();
    let tasks = Arc::new(lk_agents_net::task::TaskSet::default());
    super::super::answer_hook(hooks, call, agents, handle.api.clone(), spawner, tasks).await;
    while let Ok(event) = runtime.queue.try_recv() {
        if let VoiceRuntimeEvent::Input(VoiceEvent::Host(Sent {
            command: Command::HookDone { outcome, .. },
            ..
        })) = event
        {
            return outcome;
        }
    }
    panic!("the hook was not answered")
}

// Catches: a hook's answer translated wrongly for the driver: a `StopResponse` lost, an unchanged
// transcript sent as an edit, an `on_exit` that leaves the agent's toolsets open, a filler's
// reply misread, or a raising hook not reported as failed.
#[tokio::test]
async fn hook_answers_reach_the_driver_as_the_host_gave_them() {
    use lk_agents_voice::agent::TurnEdit;
    let hooks = Arc::new(ScriptedHooks {
        stop: true,
        user_text: Some("edited".into()),
        ..Default::default()
    });
    let (id, agent) = (HookId(1), "agent".to_owned());
    let span = |n| Some(SpanRef { trace_id: TraceId(7), span_id: SpanId(n), sampled: true });
    assert!(matches!(
        answered(&hooks, HookCall::OnEnter { id, agent: agent.clone(), scope: 0, span: span(1) })
            .await,
        HookOutcome::Returned
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnEnter { id, agent: "unknown".into(), scope: 0, span: None })
            .await,
        HookOutcome::Failed
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnExit { id, agent: agent.clone(), scope: 0, span: span(2) })
            .await,
        HookOutcome::Returned
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnExit { id, agent: "unknown".into(), scope: 0, span: None })
            .await,
        HookOutcome::Failed
    ));
    settle().await;
    assert_eq!(*hooks.toolsets.closed.lock(), 1, "on_exit left the agent's toolsets open");

    let turn = |text: &str| HookCall::OnUserTurnCompleted {
        id,
        scope: 0,
        agent: "agent".into(),
        chat_ctx: Vec::new(),
        turn: lk_agents_voice::audio_recognition::TurnId(1),
        new_transcript: text.into(),
        transcript_confidence: 0.9,
        span: span(3),
    };
    let HookOutcome::Turn(edits) = answered(&hooks, turn("original")).await else {
        panic!("not a turn")
    };
    assert!(matches!(
        edits[..],
        [TurnEdit::StopResponse, TurnEdit::UpdateUserMessage(_), TurnEdit::ReplaceItems(_)]
    ));
    let HookOutcome::Turn(edits) = answered(&hooks, turn("edited")).await else {
        panic!("not a turn")
    };
    assert!(
        !edits.iter().any(|e| matches!(e, TurnEdit::UpdateUserMessage(_))),
        "an unchanged text sent as an edit"
    );
    // Catches: a hook run outside its span, so the application's spans inside it lose their
    // parent.
    assert_eq!(*hooks.under.lock(), [span(1), span(2), span(3), span(3)], "hooks ran under spans");

    let event = lk_agents_abi::voice::events::ToolsExecutedEvent {
        function_calls: Vec::new(),
        function_call_outputs: vec![lk_agents_abi::FunctionCallOutput::new(
            "item_1",
            "call_1",
            "done",
            false,
            std::time::UNIX_EPOCH,
        )],
        handoff_required: false,
    };
    let executed =
        HookCall::FunctionToolsExecuted { id, speech: SpeechId(1), event: Box::new(event) };
    assert!(matches!(
        answered(&hooks, executed).await,
        HookOutcome::FunctionToolsExecuted { reply_required } if reply_required == [false]
    ));

    let filler = |step, source| HookCall::FillerSource {
        id,
        filler: lk_agents_voice::filler_scheduler::FillerId(1),
        step,
        source,
    };
    assert!(matches!(
        answered(&hooks, filler(0, Some(host_ref(2)))).await,
        HookOutcome::Filler { text: Some(_), spoke: false }
    ));
    assert!(matches!(
        answered(&hooks, filler(1, Some(host_ref(2)))).await,
        HookOutcome::Filler { text: None, spoke: true }
    ));
    assert!(matches!(
        answered(&hooks, filler(2, Some(host_ref(2)))).await,
        HookOutcome::Filler { text: None, spoke: false }
    ));
    assert!(matches!(
        answered(&hooks, filler(0, None)).await,
        HookOutcome::Filler { text: None, spoke: false }
    ));
    assert!(hooks.failed.lock().is_empty());

    let raising = Arc::new(ScriptedHooks { raise: true, ..Default::default() });
    // The exception reaches the core, which records it on the hook's span.
    assert!(matches!(answered(&raising, turn("x")).await, HookOutcome::Raised(_)));
    assert_eq!(*raising.failed.lock(), ["on_user_turn_completed"]);
}

struct Instructions;
impl lk_agents_abi::agent::HostInstructions for Instructions {
    fn render<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Arc<dyn lk_agents_abi::voice::speech_handle::Speech>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<String, HostError>> {
        Box::pin(async { Ok(String::new()) })
    }
}

#[tokio::test(start_paused = true)]
async fn a_handoff_refused_after_preparation_retires_its_registered_callbacks() {
    use lk_agents_abi::voice::session::VoiceSession as _;
    use lk_agents_voice::agent_session::{ActivityCommand, Answer, VoiceCommandOrigin};
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "agent".into(),
        hooks: Some(Arc::new(ScriptedHooks::default())),
        ..Default::default()
    };
    let api =
        super::super::VoiceSession::new(&env, Arc::new(super::no_room()), args, None).unwrap();
    let api = super::running(api).await;
    let (call, _) = api.call(Answer::handoff).unwrap();
    // A native close can finish between the call's admission and its frame's registration.
    api.aclose("user_initiated".into(), false).unwrap().await;
    let instructions = Arc::new(Instructions);
    let retained = Arc::downgrade(&instructions);
    let id = api.resources.register_instructions(instructions);
    let mut update =
        super::agent_update("next", lk_agents_voice::generation::Instructions::plain(""));
    update.dynamic_instructions = Some(id);
    update.toolset.host = Some(host_ref(1));
    assert!(
        api.send_frame(
            VoiceCommandOrigin::Host,
            update,
            call,
            |frame| ActivityCommand::Update(Box::new(frame)).into(),
            |_| Answer::Handoff(Err(lk_agents_voice::agent_session::HandoffRefused::SetupFailed)),
        )
        .is_err()
    );
    assert!(retained.upgrade().is_none(), "the refused callback stayed in the registry");
}

#[derive(Default)]
struct WaitingScopes(tokio::sync::Notify);
impl lk_agents_abi::llm::toolsets::ScopesPort for WaitingScopes {
    fn acquire(
        &self,
        _: u64,
        _: Vec<Arc<dyn lk_agents_abi::llm::toolsets::ToolsetHooks>>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<u64, HostError>> {
        Box::pin(async {
            self.0.notify_one();
            std::future::pending().await
        })
    }
    fn discard(&self, _: u64) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn release(&self, _: Option<u64>) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
}

struct DropCheckedInstructions {
    handoffs: Arc<tokio::sync::Mutex<()>>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}
impl Drop for DropCheckedInstructions {
    fn drop(&mut self) {
        assert!(self.handoffs.try_lock().is_ok(), "callback retired under the handoff turn");
        self.dropped.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}
impl lk_agents_abi::agent::HostInstructions for DropCheckedInstructions {
    fn render<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: Arc<dyn lk_agents_abi::voice::speech_handle::Speech>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<String, HostError>> {
        Box::pin(async { Ok(String::new()) })
    }
}

async fn session_with_hooks(hooks: Arc<ScriptedHooks>) -> super::super::VoiceSession {
    use lk_agents_abi::voice::session::VoiceSession as _;
    let env = lk_agents_net::env::ProcessEnv::current_runtime();
    let args = lk_agents_abi::voice::session::VoiceSessionArgs {
        agent: "agent".into(),
        hooks: Some(hooks),
        ..Default::default()
    };
    let api =
        super::super::VoiceSession::new(&env, Arc::new(super::no_room()), args, None).unwrap();
    super::running(api).await
}

#[tokio::test(start_paused = true)]
async fn cancelling_preparation_retires_callbacks_after_releasing_the_handoff_turn() {
    use lk_agents_abi::voice::session::VoiceSession as _;
    use lk_agents_voice::agent_session::{ActivityCommand, Answer, VoiceCommandOrigin};
    let hooks = Arc::new(ScriptedHooks::default());
    let waiting = Arc::new(WaitingScopes::default());
    assert!(hooks.toolsets.port.set(waiting.clone()).is_ok());
    let api = session_with_hooks(hooks).await;
    let (call, _) = api.call(Answer::handoff).unwrap();
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let instructions = Arc::new(DropCheckedInstructions {
        handoffs: api.host.handoffs.clone(),
        dropped: dropped.clone(),
    });
    let retained = Arc::downgrade(&instructions);
    let mut update =
        super::agent_update("next", lk_agents_voice::generation::Instructions::plain(""));
    update.dynamic_instructions = Some(api.resources.register_instructions(instructions));
    update.toolset.host = Some(host_ref(1));
    api.send_frame(
        VoiceCommandOrigin::Host,
        update,
        call,
        |frame| ActivityCommand::Update(Box::new(frame)).into(),
        |_| Answer::Handoff(Err(lk_agents_voice::agent_session::HandoffRefused::SetupFailed)),
    )
    .unwrap();
    waiting.0.notified().await;
    let clock = api.time().clock();
    api.tasks.cancel_and_wait(clock.as_ref()).await;
    assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    assert!(retained.upgrade().is_none(), "cancelled setup retained an unsubmitted callback");
    api.aclose("user_initiated".into(), false).unwrap().await;
}

#[tokio::test(start_paused = true)]
async fn closed_setup_task_admission_retires_an_unpolled_handoff_without_panicking() {
    use lk_agents_abi::voice::session::VoiceSession as _;
    use lk_agents_voice::agent_session::{ActivityCommand, Answer, VoiceCommandOrigin};
    let api = session_with_hooks(Arc::new(ScriptedHooks::default())).await;
    let (call, _) = api.call(Answer::handoff).unwrap();
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let instructions = Arc::new(DropCheckedInstructions {
        handoffs: api.host.handoffs.clone(),
        dropped: dropped.clone(),
    });
    let retained = Arc::downgrade(&instructions);
    let mut update =
        super::agent_update("next", lk_agents_voice::generation::Instructions::plain(""));
    update.dynamic_instructions = Some(api.resources.register_instructions(instructions));
    update.toolset.host = Some(host_ref(1));
    api.tasks.close();
    assert!(
        api.send_frame(
            VoiceCommandOrigin::Host,
            update,
            call,
            |frame| ActivityCommand::Update(Box::new(frame)).into(),
            |_| Answer::Handoff(Err(lk_agents_voice::agent_session::HandoffRefused::SetupFailed)),
        )
        .is_err()
    );
    assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    assert!(retained.upgrade().is_none());
    settle().await;
    api.aclose("user_initiated".into(), false).unwrap().await;
}
