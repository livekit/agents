//! The host's answers to the hooks the loop asks for, as the driver receives them.

use super::super::HostAgents;
use super::settle;
use crate::voice::Event;
use lk_agents_abi::errors::HostError;
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
            })
        } else {
            Ok(value)
        }
    }
}

impl SessionHooks for ScriptedHooks {
    fn on_enter<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn on_exit<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
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
    fn closing(&self) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn on_user_turn_completed<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
        _: Vec<lk_agents_abi::ChatItem>,
        _: &'a str,
        _: f64,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<(Option<String>, Option<Vec<lk_agents_abi::ChatItem>>, bool), HostError>,
    > {
        Box::pin(async move { self.result((self.user_text.clone(), Some(Vec::new()), self.stop)) })
    }
    fn on_user_turn_exceeded<'a>(
        &'a self,
        _: &'a lk_agents_abi::HostRef,
        _: &'a lk_agents_abi::voice::events::UserTurnExceededEvent,
    ) -> lk_agents_abi::BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async move { self.result(()) })
    }
    fn function_tools_executed(
        &self,
        _: Vec<lk_agents_abi::ChatItem>,
        outputs: Vec<lk_agents_abi::ChatItem>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Vec<bool>, HostError>> {
        Box::pin(async move { self.result(vec![false; outputs.len()]) })
    }
    fn filler_source<'a>(
        &'a self,
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
    use lk_agents_voice::agent_session::{Input, Sent};
    let (mut runtime, handle, _) = crate::voice::tests::session("hook-answers").await;
    let agents = HostAgents::default();
    agents.install("agent", 0, Some(host_ref(1)));
    let spawner =
        Arc::new(lk_agents_net::task::TokioSpawner::new(tokio::runtime::Handle::current()));
    let hooks: Arc<dyn SessionHooks> = hooks.clone();
    let tasks = Arc::new(lk_agents_net::task::TaskSet::default());
    super::super::answer_hook(hooks, call, agents, handle.api.clone(), spawner, tasks).await;
    while let Ok(event) = runtime.queue.try_recv() {
        if let Event::Input(Input::Host(Sent {
            command: Command::HookDone { outcome, .. }, ..
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
    assert!(matches!(
        answered(&hooks, HookCall::OnEnter { id, agent: agent.clone(), scope: 0 }).await,
        HookOutcome::Returned
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnEnter { id, agent: "unknown".into(), scope: 0 }).await,
        HookOutcome::Failed
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnExit { id, agent: agent.clone(), scope: 0 }).await,
        HookOutcome::Returned
    ));
    assert!(matches!(
        answered(&hooks, HookCall::OnExit { id, agent: "unknown".into(), scope: 0 }).await,
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
    assert!(matches!(answered(&raising, turn("x")).await, HookOutcome::Failed));
    assert_eq!(*raising.failed.lock(), ["on_user_turn_completed"]);
}
