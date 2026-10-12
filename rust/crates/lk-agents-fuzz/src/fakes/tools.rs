//! Tool bodies: every call the product hands to the runner is an invocation with the actual
//! `call_id`, name, speech and step; the body waits for whatever the case delivers.
use std::sync::Arc;

use lk_agents_abi::FunctionCall;
use lk_agents_voice::{
    agent_activity::scheduling::SpeechId,
    agent_session::{Origin, SpeechCommand},
    tool_executor::{ToolAnswer, ToolRunner},
};
use tokio::sync::oneshot;

use super::{Control, Registry};
use crate::{
    input::{AgentText, Input, ToolAnswer as Answer},
    record::{Detail, Kind},
};

pub struct FakeTools {
    registry: Arc<Registry>,
}

impl FakeTools {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        Arc::new(FakeTools { registry })
    }
}

/// What the case tells a running body to do: answer, or first wait for the agent to go idle --
/// `ctx.wait_for_idle()` -- and answer once it has.
pub enum Body {
    Answer(ToolAnswer),
    AfterIdle(ToolAnswer),
}

#[async_trait::async_trait]
impl ToolRunner for FakeTools {
    async fn run(&self, call: FunctionCall, speech: SpeechId, _step: u32) -> ToolAnswer {
        let (sender, receiver) = oneshot::channel();
        self.registry.register(
            Kind::Tool,
            Detail::Tool {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                speech: speech.0,
            },
            Control::Tool(sender),
        );
        match receiver.await {
            Ok(Body::Answer(answer)) => answer,
            Ok(Body::AfterIdle(answer)) => {
                // The body asks from its own origin, as the binding stamps a tool's commands;
                // the driver's answer -- idle, or refused by the close -- ends the wait either way.
                if let Some(api) = self.registry.api() {
                    let (done, idle) = oneshot::channel();
                    let origin = Origin::Tool { call_id: call.call_id.clone() };
                    let asked = api
                        .send_from(origin, SpeechCommand::WaitForIdle { hold: None, done }.into())
                        .is_ok();
                    if asked {
                        let _ = idle.await;
                    }
                }
                answer
            }
            Err(_) => ToolAnswer::Cancelled,
        }
    }
}

/// One answer to a registered tool body. False when the body was already cancelled.
pub(super) fn deliver(sender: oneshot::Sender<Body>, what: &Input) -> bool {
    use lk_agents_voice::tool_executor::ToolRaise;
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
