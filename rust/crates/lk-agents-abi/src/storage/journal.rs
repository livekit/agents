//! The journal's events (docs/internals/storage.md#journal): one typed record per event type,
//! stored as its `type` and JSON payload.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use serde_json::Value;

#[cfg(feature = "serde")]
use super::thread::RunOutcome;

/// One journal entry's event, as the core or storage that owns it records it.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum JournalEvent {
    RunStarted(RunStarted),
    RunEnded(RunEnded),
    Usage(#[serde(deserialize_with = "stored_usage")] crate::usage::Usage),
    /// An item entered the history.
    ConversationItemAdded {
        item: Box<crate::livekit_protocol::agent::chat_context::ChatItem>,
    },
    ToolExecutionUpdated(ToolExecution),
    Error {
        error: String,
    },
}

#[cfg(feature = "serde")]
impl JournalEvent {
    /// The stored `(type, payload)`.
    pub fn into_row(self) -> (String, Value) {
        let mut row = match serde_json::to_value(self) {
            Ok(Value::Object(row)) => row,
            _ => unreachable!("a journal event serializes as its tagged object"),
        };
        let r#type = row.remove("type").and_then(|t| t.as_str().map(str::to_owned));
        (r#type.unwrap_or_default(), row.remove("payload").unwrap_or(Value::Null))
    }

    /// A stored entry; an unknown type or a payload of the wrong shape is refused.
    pub fn from_row(r#type: &str, payload: Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(serde_json::json!({"type": r#type, "payload": payload}))
    }
}

/// A stored `usage` payload: the record, or a row written before it existed
/// (`{requests, input_tokens, output_tokens, tool_calls}`), read as one unidentified LLM entry.
#[cfg(feature = "serde")]
fn stored_usage<'de, D: serde::Deserializer<'de>>(d: D) -> Result<crate::usage::Usage, D::Error> {
    use crate::usage::{LlmUsage, ModelUsage, TokenUsage, Usage};
    use serde::de::Error as _;
    #[derive(Deserialize, Default)]
    #[serde(default)]
    struct Legacy {
        requests: u64,
        input_tokens: u64,
        output_tokens: u64,
        tool_calls: u64,
    }
    let value = Value::deserialize(d)?;
    if value.get("models").is_some() {
        return serde_json::from_value(value).map_err(D::Error::custom);
    }
    let legacy: Legacy = serde_json::from_value(value).map_err(D::Error::custom)?;
    let tokens = TokenUsage {
        input_tokens: legacy.input_tokens,
        output_tokens: legacy.output_tokens,
        ..TokenUsage::default()
    };
    let llm = LlmUsage { requests: legacy.requests, tokens: tokens.clone(), ..LlmUsage::default() };
    Ok(Usage {
        models: if legacy.requests == 0 { Vec::new() } else { vec![ModelUsage::Llm(llm)] },
        tokens,
        requests: legacy.requests,
        tool_calls: legacy.tool_calls,
        settled: true,
        ..Usage::default()
    })
}

/// `run_started`: a run began.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RunStarted {
    #[serde(default)]
    pub task_id: Option<String>,
    /// The answer stream, the run's task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    /// `false` for a `concurrent` run, which holds no lease.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclusive: Option<bool>,
}

/// `run_ended`: how the run ended.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEnded {
    pub outcome: RunOutcome,
    #[serde(default)]
    pub reason: Option<String>,
}

/// `tool_execution_updated`: one call's start, a progress update from `ctx.update()`, its end.
#[crate::host(record(module = "storage_turn", extract, emit))]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ToolExecution {
    pub phase: ToolPhase,
    #[host(default = None)]
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub call_id: Option<String>,
    #[host(default = None)]
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub name: Option<String>,
    #[host(default = None)]
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub is_error: Option<bool>,
    /// The progress message of an `update`.
    #[host(default = None)]
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub text: Option<String>,
}

#[crate::host(record(module = "storage_turn", extract, emit))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ToolPhase {
    #[host(rename = "started")]
    Started,
    #[host(rename = "update")]
    Update,
    #[host(rename = "ended")]
    Ended,
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;
    use serde_json::json;

    /// docs/internals/storage.md#journal: every event round-trips under its documented type and
    /// payload; a malformed payload is refused.
    #[test]
    fn every_journal_event_round_trips_its_documented_row() {
        let tool = ToolExecution {
            phase: ToolPhase::Ended,
            call_id: Some("call_1".into()),
            name: Some("book".into()),
            is_error: Some(false),
            text: None,
        };
        let usage = crate::usage::Usage { requests: 1, tool_calls: 1, ..Default::default() };
        let rows = [
            (JournalEvent::RunStarted(RunStarted::default()), "run_started"),
            (
                JournalEvent::RunEnded(RunEnded { outcome: RunOutcome::Canceled, reason: None }),
                "run_ended",
            ),
            (JournalEvent::Usage(usage), "usage"),
            (
                JournalEvent::ConversationItemAdded { item: Box::default() },
                "conversation_item_added",
            ),
            (JournalEvent::ToolExecutionUpdated(tool), "tool_execution_updated"),
            (JournalEvent::Error { error: "boom".into() }, "error"),
        ];
        for (event, documented) in rows {
            let (r#type, payload) = event.clone().into_row();
            assert_eq!(r#type, documented);
            assert_eq!(JournalEvent::from_row(&r#type, payload).unwrap(), event);
        }
        let (_, payload) = JournalEvent::ToolExecutionUpdated(ToolExecution {
            phase: ToolPhase::Update,
            call_id: None,
            name: None,
            is_error: None,
            text: Some("checking".into()),
        })
        .into_row();
        assert_eq!(payload, json!({"phase": "update", "text": "checking"}));
        assert!(JournalEvent::from_row("run_ended", json!({"outcome": "maybe"})).is_err());
    }

    /// usage.md "Removing duplicate contracts": a row stored before `Usage` existed reads as one
    /// unidentified LLM entry, with no old field names in the returned record.
    #[test]
    fn a_legacy_usage_row_reads_as_one_unidentified_llm_entry() {
        use crate::usage::{LlmUsage, ModelUsage, TokenUsage};
        let row = json!({"requests": 2, "input_tokens": 100, "output_tokens": 30, "tool_calls": 1});
        let JournalEvent::Usage(usage) = JournalEvent::from_row("usage", row).unwrap() else {
            panic!("a usage row is a usage event");
        };
        let tokens = TokenUsage { input_tokens: 100, output_tokens: 30, ..TokenUsage::default() };
        let llm = LlmUsage { requests: 2, tokens: tokens.clone(), ..LlmUsage::default() };
        assert_eq!(usage.models, vec![ModelUsage::Llm(llm)]);
        assert_eq!((&usage.tokens, usage.requests, usage.tool_calls), (&tokens, 2, 1));
        assert!(usage.settled);
        let (_, payload) = JournalEvent::Usage(usage).into_row();
        assert!(payload.get("models").is_some(), "re-stored in the record's shape: {payload}");
    }
}
