//! Tool bodies: every call the product hands to the runner is an invocation with the actual
//! `call_id`, name, speech and step; the body waits for whatever the case delivers. The runner is
//! the host's (`HostToolRunner`), classifying what the body did, as a Python tool's runner does.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use lk_agents_abi::{FunctionCall, errors::HostError, llm::tool_executor::Approval};
use lk_agents_voice::speech_handle::SpeechHandle;
use lk_agents_voice::tool_executor::{HostToolRunner, ToolOutcome, ToolReport};
use tokio::sync::oneshot;

use super::{Control, Registry, Slot};
use crate::{
    input::{AgentText, Input, ToolAnswer as Answer},
    record::{Detail, Kind},
};

/// The tool the runs declare with `needs_approval=True`.
pub const APPROVAL_TOOL: &str = "transfer";

pub struct FakeTools {
    registry: Arc<Registry>,
    /// The bodies that answered `Detached` and run on, by call id: what a cancel ends.
    detached: Mutex<BTreeMap<String, Slot>>,
}

impl FakeTools {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeTools { registry, detached: Mutex::default() })
    }
}

/// What the case tells a running body to do: answer, or first wait for the agent to go idle --
/// `ctx.wait_for_idle()` -- and answer once it has.
pub enum Body {
    Answer(ToolOutcome),
    AfterIdle(ToolOutcome),
    /// `await ctx.speech_handle` first: refused, or the body waits on its own speech forever.
    AwaitOwnSpeech(ToolOutcome),
}

#[async_trait::async_trait]
impl HostToolRunner for FakeTools {
    async fn run(
        &self,
        call: &FunctionCall,
        speech: &lk_agents_abi::CoreObject,
        granted: bool,
    ) -> Result<ToolReport, HostError> {
        let handle = Arc::clone(&speech.0)
            .downcast::<SpeechHandle>()
            .ok()
            .filter(|h| h.created().is_some())
            .expect("a tool runs in a created speech");
        // `needs_approval=True`: not granted, the call is its read-back, and the body never runs.
        if call.name == APPROVAL_TOOL && !granted {
            // The runner saw the call: one body opened and ended at once.
            let detail = Detail::Tool {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                speech: handle.id().0,
            };
            let (sender, _) = oneshot::channel();
            let slot = self.registry.register(Kind::Tool, detail, Control::Tool(sender));
            self.registry.mark_closed(slot);
            let question =
                format!("Please confirm: {} ({}). Yes or no?", call.name, call.arguments);
            let outcome = ToolOutcome::InputRequired { question, verbatim: true, tool: None };
            return Ok(ToolReport { outcome, approval: Some(Approval::Asked), transition: None });
        }
        let mut outcome = self.answer(call, &handle).await;
        // A question asked as written names its own tool to take the answer
        // (`InputRequired(question, tool=..., verbatim=True)`); one in the agent's words does not.
        if let ToolOutcome::InputRequired { verbatim: true, tool, .. } = &mut outcome {
            *tool = Some(call.name.clone());
        }
        Ok(ToolReport { outcome, approval: granted.then_some(Approval::Granted), transition: None })
    }

    fn cancel(&self, call_id: &str) {
        if let Some(slot) = self.detached.lock().unwrap().remove(call_id) {
            self.registry.mark_closed(slot);
        }
    }
}

impl FakeTools {
    async fn answer(&self, call: &FunctionCall, handle: &SpeechHandle) -> ToolOutcome {
        let speech = handle.id().0;
        let (sender, receiver) = oneshot::channel();
        let slot = self.registry.register(
            Kind::Tool,
            Detail::Tool { call_id: call.call_id.clone(), name: call.name.clone(), speech },
            Control::Tool(sender),
        );
        match receiver.await {
            Ok(Body::Answer(answer)) => {
                if let ToolOutcome::Update { .. } = answer {
                    self.detached.lock().unwrap().insert(call.call_id.clone(), slot);
                }
                answer
            }
            Ok(Body::AfterIdle(answer)) => {
                // `async with ctx.foreground():` in the body: idle asked from the tool's own
                // origin (so the driver can tell it apart from its speech), the floor held, and
                // released on the way out. Refused by the close, the wait ends all the same.
                let origin = format!("tool:{}", call.call_id);
                if let Some(session) = self.registry.session()
                    && let Ok(hold) = session.wait_for_idle_and_hold(Some(&origin))
                    && let Ok(hold) = hold.await
                {
                    session.release_idle_hold(hold);
                }
                answer
            }
            Ok(Body::AwaitOwnSpeech(answer)) => {
                let origin = format!("tool:{}", call.call_id);
                let _ = handle.wait_for_playout_for_host(Some(origin)).await;
                answer
            }
            Err(_) => ToolOutcome::Cancelled,
        }
    }
}

/// One answer to a registered tool body. False when the body was already cancelled.
pub(super) fn deliver(sender: oneshot::Sender<Body>, what: &Input) -> bool {
    let Input::ToolAnswered { answer, text: AgentText(text), .. } = what else { return false };
    match answer {
        Answer::AfterIdle => sender.send(Body::AfterIdle(ToolOutcome::Nothing)).is_ok(),
        Answer::AwaitOwnSpeech => sender.send(Body::AwaitOwnSpeech(ToolOutcome::Nothing)).is_ok(),
        _ => sender.send(Body::Answer(outcome_of(*answer, text))).is_ok(),
    }
}

/// What the runner reports for a body that answered `answer` with `text`, as a Python tool's
/// runner classifies the value it returned or raised.
pub fn outcome_of(answer: Answer, text: &str) -> ToolOutcome {
    let text = text.to_owned();
    let value = |reply_required, verbatim| ToolOutcome::Value {
        text: text.clone(),
        reply_required,
        verbatim,
        more: None,
    };
    match answer {
        Answer::Return | Answer::AfterIdle | Answer::AwaitOwnSpeech => ToolOutcome::Nothing,
        Answer::Value => value(true, false),
        Answer::Recorded => value(false, false),
        Answer::Verbatim => value(true, true),
        Answer::Error => ToolOutcome::ToolError(text),
        Answer::StopResponse => ToolOutcome::StopResponse,
        Answer::Cancelled => ToolOutcome::Cancelled,
        Answer::Detached => ToolOutcome::Update { message: text.clone(), raw: text },
        Answer::Completed => ToolOutcome::Completed { text, verbatim: false },
        Answer::Failed => ToolOutcome::Failed(text),
        Answer::Rejected => ToolOutcome::Rejected(text),
        Answer::Ask => ToolOutcome::InputRequired { question: text, verbatim: false, tool: None },
        Answer::AskVerbatim => {
            ToolOutcome::InputRequired { question: text, verbatim: true, tool: None }
        }
        Answer::Working => ToolOutcome::Working(Some(text)),
    }
}
