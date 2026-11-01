//! Public event envelopes, text snapshots and child-result presentation acknowledgments.

use super::View;
use lk_agents_abi::agent::run::{StreamEvent, ToolResultStatus};
use lk_agents_abi::agent::stream::*;
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct Envelope {
    pub sequence: Option<u64>,
    pub agent_name: String,
    pub invocation_id: String,
    pub frame_id: Option<String>,
    pub event: StreamEvent,
}

#[derive(Clone)]
pub enum Item {
    Event(Box<PresentedEvent>),
    Text(String),
}

#[derive(Default)]
pub struct Projection {
    sequence: u64,
    message: Option<String>,
    text: String,
    presented: BTreeMap<String, String>,
    pending: Option<Pending>,
}

struct Pending {
    consumer: u64,
    envelope: Envelope,
    view: View,
}

impl Projection {
    pub fn select(&mut self, envelope: Envelope, view: View, consumer: u64) -> Option<Item> {
        let saved = envelope.clone();
        let item = self.render(envelope, view);
        if item.is_some() {
            self.pending = Some(Pending { consumer, envelope: saved, view });
        } else {
            self.sequence = saved.sequence.unwrap_or(self.sequence + 1);
        }
        item
    }
    pub fn retry(&mut self, view: View, consumer: u64) -> Option<Item> {
        let pending = self.pending.take()?;
        self.select(pending.envelope, view, consumer)
    }
    pub fn acknowledge(&mut self, consumer: u64, accepted: bool) {
        if !accepted || !self.pending.as_ref().is_some_and(|pending| pending.consumer == consumer) {
            return;
        }
        let pending = self.pending.take().expect("delivery identity checked");
        self.sequence = pending.envelope.sequence.unwrap_or(self.sequence + 1);
        if let StreamEvent::TextDelta { message_id, delta } = pending.envelope.event {
            self.presented.entry(message_id.clone()).or_default().push_str(&delta);
            if matches!(pending.view, View::Text { delta: false }) {
                if self.message.as_ref() != Some(&message_id) {
                    self.message = Some(message_id);
                    self.text.clear();
                }
                self.text.push_str(&delta);
            }
        }
    }
    fn render(&self, envelope: Envelope, view: View) -> Option<Item> {
        let Envelope { sequence, agent_name, invocation_id, frame_id, event } = envelope;
        let sequence = sequence.unwrap_or(self.sequence + 1);
        if view == View::Result {
            return None;
        }
        if let StreamEvent::TextDelta { message_id, delta } = &event {
            if let View::Text { delta: fragments } = view {
                let text = if fragments || self.message.as_ref() != Some(message_id) {
                    delta.clone()
                } else {
                    let mut text = self.text.clone();
                    text.push_str(delta);
                    text
                };
                return Some(Item::Text(text));
            }
        } else if matches!(view, View::Text { .. }) {
            return None;
        }
        Some(Item::Event(Box::new(match event {
            StreamEvent::TextDelta { message_id, delta } => {
                PresentedEvent::TextDelta(TextDeltaEvent {
                    sequence,
                    agent_name,
                    invocation_id,
                    frame_id,
                    message_id,
                    delta,
                })
            }
            StreamEvent::TextEnd { message_id, interrupted } => {
                PresentedEvent::TextEnd(TextEndEvent {
                    sequence,
                    agent_name,
                    invocation_id,
                    frame_id,
                    message_id,
                    interrupted,
                })
            }
            StreamEvent::ToolCall { call_id, tool_name, arguments } => {
                PresentedEvent::ToolCall(ToolCallEvent {
                    sequence,
                    agent_name,
                    invocation_id,
                    frame_id,
                    call_id,
                    tool_name,
                    arguments: if arguments.is_null() { serde_json::json!({}) } else { arguments },
                })
            }
            StreamEvent::ToolResult { call_id, tool_name, status, output } => {
                let failed = status == ToolResultStatus::Failed;
                PresentedEvent::ToolResult(ToolResultEvent {
                    sequence,
                    agent_name,
                    invocation_id,
                    frame_id,
                    call_id,
                    tool_name,
                    outcome: status,
                    output: if failed {
                        serde_json::Value::Null
                    } else {
                        serde_json::Value::String(output.clone())
                    },
                    error: failed.then_some(output),
                })
            }
            StreamEvent::Progress { call_id, tool_name, message } => {
                PresentedEvent::Progress(ProgressEvent {
                    sequence,
                    agent_name,
                    invocation_id,
                    frame_id,
                    call_id: Some(call_id),
                    tool_name: Some(tool_name),
                    message,
                })
            }
        })))
    }
    pub fn release(&mut self) -> BTreeMap<String, String> {
        std::mem::take(&mut self.presented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text(message: &str, delta: &str) -> Envelope {
        Envelope {
            sequence: None,
            agent_name: "agent".into(),
            invocation_id: "run".into(),
            frame_id: None,
            event: StreamEvent::TextDelta { message_id: message.into(), delta: delta.into() },
        }
    }
    #[test]
    fn snapshots_reset_per_message_and_only_presented_fragments_are_acknowledged() {
        let mut p = Projection::default();
        let snapshot = View::Text { delta: false };
        assert!(
            matches!(p.select(text("a", "hello"), snapshot, 1), Some(Item::Text(t)) if t == "hello")
        );
        p.acknowledge(1, true);
        assert!(
            matches!(p.select(text("a", " world"), snapshot, 1), Some(Item::Text(t)) if t == "hello world")
        );
        p.acknowledge(1, true);
        assert!(p.select(text("a", " hidden"), View::Result, 1).is_none());
        assert!(
            matches!(p.select(text("b", "next"), snapshot, 1), Some(Item::Text(t)) if t == "next")
        );
        p.acknowledge(1, true);
        assert_eq!(
            p.release(),
            BTreeMap::from([("a".into(), "hello world".into()), ("b".into(), "next".into())])
        );
        assert!(p.release().is_empty());
    }
}
