//! Tool bodies: every call the product hands to the runner is an invocation with the actual
//! `call_id`, name, speech and step; the body waits for whatever the case delivers. The runner is
//! the host's (`HostToolRunner`), answering with the marshal's report, as a Python tool does.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use lk_agents_abi::{
    FunctionCall,
    errors::HostError,
    llm::tool_executor::{RawToolAnswer, ToolAnswerKind, ToolRaiseKind},
};
use lk_agents_voice::tool_executor::{HostToolRunner, ToolAnswer, ToolRaise};
use tokio::sync::oneshot;

use super::{Control, Registry, Slot};
use crate::{
    input::{AgentText, Input, ToolAnswer as Answer},
    record::{Detail, Kind},
};

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
    Answer(ToolAnswer),
    AfterIdle(ToolAnswer),
}

#[async_trait::async_trait]
impl HostToolRunner for FakeTools {
    async fn run(
        &self,
        call: &FunctionCall,
        speech: u64,
        _step: u32,
    ) -> Result<RawToolAnswer, HostError> {
        Ok(raw(self.answer(call, speech).await))
    }

    fn cancel(&self, call_id: &str) {
        if let Some(slot) = self.detached.lock().unwrap().remove(call_id) {
            self.registry.mark_closed(slot);
        }
    }
}

impl FakeTools {
    async fn answer(&self, call: &FunctionCall, speech: u64) -> ToolAnswer {
        let (sender, receiver) = oneshot::channel();
        let slot = self.registry.register(
            Kind::Tool,
            Detail::Tool { call_id: call.call_id.clone(), name: call.name.clone(), speech },
            Control::Tool(sender),
        );
        match receiver.await {
            Ok(Body::Answer(answer)) => {
                if let ToolAnswer::Detached { .. } = answer {
                    self.detached.lock().unwrap().insert(call.call_id.clone(), slot);
                }
                answer
            }
            Ok(Body::AfterIdle(answer)) => {
                // `await session.wait_for_idle()` in the body: the driver's answer, idle or
                // refused by the close, ends the wait either way.
                if let Some(idle) = self.registry.session().and_then(|s| s.wait_for_idle(None).ok())
                {
                    let _ = idle.await;
                }
                answer
            }
            Err(_) => ToolAnswer::Cancelled,
        }
    }
}

/// The marshal's report of an answer, as `_ToolExecutor` writes it.
fn raw(answer: ToolAnswer) -> RawToolAnswer {
    let mut raw = RawToolAnswer {
        kind: String::new(),
        value: None,
        agents: Vec::new(),
        error: None,
        message: None,
        raw: None,
    };
    let kind = match answer {
        ToolAnswer::Returned { value, agents } => {
            (raw.value, raw.agents) = (value, agents);
            ToolAnswerKind::Returned
        }
        ToolAnswer::Raised(raised) => {
            let (kind, message) = match raised {
                ToolRaise::ToolError(message) => (ToolRaiseKind::ToolError, Some(message)),
                ToolRaise::StopResponse => (ToolRaiseKind::StopResponse, None),
                ToolRaise::Internal => (ToolRaiseKind::Internal, None),
            };
            (raw.error, raw.message) = (Some(kind.as_str().to_owned()), message);
            ToolAnswerKind::Raised
        }
        ToolAnswer::Invalid => ToolAnswerKind::Invalid,
        ToolAnswer::Cancelled => ToolAnswerKind::Cancelled,
        ToolAnswer::Detached { message, raw: text } => {
            (raw.message, raw.raw) = (Some(message), Some(text));
            ToolAnswerKind::Detached
        }
    };
    raw.kind = kind.as_str().to_owned();
    raw
}

/// One answer to a registered tool body. False when the body was already cancelled.
pub(super) fn deliver(sender: oneshot::Sender<Body>, what: &Input) -> bool {
    let Input::ToolAnswered { answer, text: AgentText(text), .. } = what else { return false };
    if let Answer::AfterIdle = answer {
        let returned = ToolAnswer::Returned { value: None, agents: vec![] };
        return sender.send(Body::AfterIdle(returned)).is_ok();
    }
    let answer = match answer {
        Answer::Return => ToolAnswer::Returned { value: None, agents: vec![] },
        Answer::Value => ToolAnswer::Returned {
            value: Some(serde_json::Value::String(text.clone())),
            agents: vec![],
        },
        Answer::Error => ToolAnswer::Raised(ToolRaise::ToolError(text.clone())),
        Answer::StopResponse => ToolAnswer::Raised(ToolRaise::StopResponse),
        Answer::Cancelled => ToolAnswer::Cancelled,
        Answer::Detached => ToolAnswer::Detached { message: text.clone(), raw: text.clone() },
        Answer::AfterIdle => unreachable!("answered above"),
    };
    sender.send(Body::Answer(answer)).is_ok()
}
