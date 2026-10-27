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
use lk_agents_voice::speech_handle::SpeechHandle;
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
    /// `await ctx.speech_handle` first: refused, or the body waits on its own speech forever.
    AwaitOwnSpeech(ToolAnswer),
}

#[async_trait::async_trait]
impl HostToolRunner for FakeTools {
    async fn run(
        &self,
        call: &FunctionCall,
        speech: &lk_agents_abi::CoreObject,
        _step: u32,
    ) -> Result<RawToolAnswer, HostError> {
        let handle = Arc::clone(&speech.0)
            .downcast::<SpeechHandle>()
            .ok()
            .filter(|h| h.created().is_some())
            .expect("a tool runs in a created speech");
        Ok(raw(self.answer(call, &handle).await))
    }

    fn cancel(&self, call_id: &str) {
        if let Some(slot) = self.detached.lock().unwrap().remove(call_id) {
            self.registry.mark_closed(slot);
        }
    }
}

impl FakeTools {
    async fn answer(&self, call: &FunctionCall, handle: &SpeechHandle) -> ToolAnswer {
        let speech = handle.id().0;
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
            Err(_) => ToolAnswer::Cancelled,
        }
    }
}

/// The marshal's report of an answer, as `_ToolExecutor` writes it.
fn raw(answer: ToolAnswer) -> RawToolAnswer {
    let mut raw =
        RawToolAnswer { kind: String::new(), value: None, error: None, message: None, raw: None };
    let kind = match answer {
        ToolAnswer::Returned { value } => {
            raw.value = value;
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
    let returned = ToolAnswer::Returned { value: None };
    match answer {
        Answer::AfterIdle => return sender.send(Body::AfterIdle(returned)).is_ok(),
        Answer::AwaitOwnSpeech => return sender.send(Body::AwaitOwnSpeech(returned)).is_ok(),
        _ => {}
    }
    let answer = match answer {
        Answer::Return => ToolAnswer::Returned { value: None },
        Answer::Value => {
            ToolAnswer::Returned { value: Some(serde_json::Value::String(text.clone())) }
        }
        Answer::Error => ToolAnswer::Raised(ToolRaise::ToolError(text.clone())),
        Answer::StopResponse => ToolAnswer::Raised(ToolRaise::StopResponse),
        Answer::Cancelled => ToolAnswer::Cancelled,
        Answer::Detached => ToolAnswer::Detached { message: text.clone(), raw: text.clone() },
        Answer::AfterIdle | Answer::AwaitOwnSpeech => unreachable!("answered above"),
    };
    sender.send(Body::Answer(answer)).is_ok()
}
