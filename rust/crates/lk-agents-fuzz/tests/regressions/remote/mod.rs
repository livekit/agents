//! A voice frontend delegating to a remote A2A peer: the real delegation, A2A client and
//! exchange executors over the scripted peer at the socket boundary.

mod delegation;

use std::collections::BTreeSet;

use lk_agents_fuzz::check::{Failure, check_all};
use lk_agents_fuzz::drive::{Run, Setup};
use lk_agents_fuzz::fakes::a2a_peer::PeerScript;
use lk_agents_fuzz::fakes::remote_tool::REMOTE_TOOL;
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolArgs, UserText};
use lk_agents_fuzz::record::{Detail, Kind, RemoteStep};

/// The frontend: a text-only pipeline whose one tool is the remote desk.
fn frontend(script: PeerScript, routing: bool) -> Setup {
    Setup {
        tts: false,
        tools: vec![REMOTE_TOOL.into()],
        input_routing: routing,
        remote: Some(script),
        ..Setup::default()
    }
}

/// The person on the call and the frontend model: each new model request is answered in turn.
/// A request offering the remote call router or the response recogniser gets `verdict` and
/// `recognised`; the frontend says a line, or calls the desk when `next_call` holds a request.
struct Call {
    answered: BTreeSet<u32>,
    pub next_call: Option<String>,
    pub verdict: &'static str,
    pub recognised: &'static str,
}

impl Call {
    fn new() -> Self {
        Call {
            answered: BTreeSet::new(),
            next_call: None,
            verdict: r#"{"action":"respond","candidate":0,"response":0}"#,
            recognised: r#"{"candidates":[0],"unclear":false}"#,
        }
    }

    /// Let `ms` of the call pass in small steps, answering every model request as it opens.
    async fn advance(&mut self, run: &mut Run, ms: u64) {
        let mut left = ms;
        loop {
            for chat in run.record.live().of(Kind::Chat) {
                if !self.answered.insert(chat) {
                    continue;
                }
                let tools = match run.record.resource(Kind::Chat, chat).map(|r| r.detail().clone()) {
                    Some(Detail::Chat { tools, .. }) => tools,
                    _ => continue,
                };
                let at = Ref::new(chat);
                let offers = |name: &str| tools.iter().any(|t| t == name);
                let first = if offers(lk_agents_core::delegation::remote::routing::TOOL) {
                    Input::LlmToolCall {
                        chat: at,
                        name: lk_agents_core::delegation::remote::routing::TOOL.into(),
                        args: ToolArgs(self.verdict.into()),
                    }
                } else if offers(lk_agents_core::session::input_routing::TOOL) {
                    Input::LlmToolCall {
                        chat: at,
                        name: lk_agents_core::session::input_routing::TOOL.into(),
                        args: ToolArgs(self.recognised.into()),
                    }
                } else if let Some(request) = self.next_call.take() {
                    Input::LlmToolCall {
                        chat: at,
                        name: REMOTE_TOOL.into(),
                        args: ToolArgs(serde_json::json!({ "request": request }).to_string()),
                    }
                } else {
                    Input::LlmText { chat: at, text: AgentText("Okay.".into()) }
                };
                run.apply(&first).await;
                run.apply(&Input::LlmEnd { chat: at }).await;
            }
            if left == 0 {
                return;
            }
            let step = left.min(50);
            left -= step;
            run.apply(&Input::Advance { ms: Millis(step) }).await;
        }
    }

    /// The person says `words`, as a committed turn the frontend answers.
    async fn say(&mut self, run: &mut Run, words: &str) {
        let input = Input::GenerateReply {
            user_input: Some(UserText(words.into())),
            instructions: None,
            allow: None,
        };
        run.apply(&input).await;
    }

    /// The frontend is asked for a reply with nothing new from the person.
    async fn prompt(&mut self, run: &mut Run) {
        let input = Input::GenerateReply { user_input: None, instructions: None, allow: None };
        run.apply(&input).await;
    }

    /// Hang up: close without draining, wait for the exit, then let cleanup land.
    async fn hang_up(&mut self, run: &mut Run) -> Vec<Failure> {
        run.apply(&Input::Close { drain: false }).await;
        self.advance(run, 5_000).await;
        run.apply(&Input::AwaitExit {}).await;
        run.apply(&Input::Advance { ms: Millis(20_000) }).await;
        check_all(&run.record).0
    }
}

/// What the peer and the remote bodies observed, in order.
fn steps(run: &Run) -> Vec<RemoteStep> {
    run.record.remote().iter().map(|(_, step)| step.clone()).collect()
}

/// The failures of the remote contracts only.
fn remote_failures(failures: &[Failure]) -> Vec<&Failure> {
    failures.iter().filter(|f| f.contract.starts_with("remote_")).collect()
}
