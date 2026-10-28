//! The fake handler: an application's `a2a_session` handler as the turn sees it through the host
//! stream. It reaches a few checkpoints, yields once told to, and ends with a text answer, a
//! conversation tool's question (its call and placeholder output in the new items, as an agent
//! run records them) or a raise.

use fastrand::Rng;
use lk_agents_abi::livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use lk_agents_abi::livekit_protocol::agent::{FunctionCall, FunctionCallOutput};
use lk_agents_abi::storage::thread::{ItemRow, timestamp};
use lk_agents_core::a2a::task::{Handle, HandlerEvent};
use lk_agents_core::a2a::{HandlerOutput, PendingCall, Verdict};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ending {
    Text,
    Ask,
    Raise,
}

/// One handler invocation.
#[derive(Debug, Clone)]
pub struct Handler {
    task_id: String,
    message: ItemRow,
    run_id: String,
    checkpoints: u8,
    ending: Ending,
    yielded: bool,
}

impl Handler {
    pub fn new(handle: &Handle, run_id: &str, rng: &mut Rng) -> Self {
        let ending = match rng.u8(0..10) {
            0..=5 => Ending::Text,
            6..=8 => Ending::Ask,
            _ => Ending::Raise,
        };
        Self {
            task_id: handle.task_id.clone(),
            message: handle.message.clone(),
            run_id: run_id.to_owned(),
            checkpoints: rng.u8(0..4),
            ending,
            yielded: false,
        }
    }

    /// The handler's next event, after `verdict` answered its last checkpoint.
    pub fn next(&mut self, verdict: Option<Verdict>) -> HandlerEvent {
        if verdict == Some(Verdict::Yield) {
            self.yielded = true;
        }
        if !self.yielded && self.checkpoints > 0 {
            self.checkpoints -= 1;
            return HandlerEvent::Checkpoint;
        }
        HandlerEvent::Output(Box::new(self.output()))
    }

    fn output(&self) -> HandlerOutput {
        let mut out = HandlerOutput { interrupted: self.yielded, ..HandlerOutput::default() };
        if self.yielded {
            return out;
        }
        match self.ending {
            Ending::Text => out.output = Some(format!("answer for {}", self.task_id).into()),
            Ending::Raise => out.error = Some("RuntimeError: the tool failed".into()),
            Ending::Ask => {
                let call_id = format!("call_{}", self.run_id);
                let at = self.message.created_at();
                let item = |kind| ChatItem { item: Some(kind) };
                out.new_items = vec![
                    self.message.payload.clone(),
                    item(chat_item::Item::FunctionCall(FunctionCall {
                        id: format!("item_{}_call", self.run_id),
                        call_id: call_id.clone(),
                        name: "ask_day".into(),
                        arguments: "{}".into(),
                        created_at: Some(timestamp(at + 1e-6)),
                        ..Default::default()
                    })),
                    item(chat_item::Item::FunctionCallOutput(FunctionCallOutput {
                        id: format!("item_{}_wait", self.run_id),
                        call_id: call_id.clone(),
                        name: "ask_day".into(),
                        output: "waiting for the person's answer to: which day?".into(),
                        is_error: false,
                        created_at: Some(timestamp(at + 2e-6)),
                    })),
                ];
                out.input_required = Some("which day?".into());
                out.pending_call =
                    Some(PendingCall { name: "ask_day".into(), call_id, answer_tool: None });
            }
        }
        out
    }
}
