//! One extract target per `AgentSessionEvent` arm forwarded to `encode_wire`. The proto mapping lives
//! in `lk-agents-voice/src/remote_session.rs`.

// The generated module globs this one to resolve field types.
pub use crate::ChatItem;
pub use crate::voice::amd::AmdCategory;
pub use crate::voice::events::{AgentState, UserState};
// The arms `events.rs` already declares, read back under `kind` (`#[binding(read_tag)]`).
pub use crate::voice::events::{
    AgentFalseInterruption, AgentStateChanged, EotPrediction, ItemAddedWire, ToolCallEnded,
    ToolCallUpdated, ToolReplyUpdated, UserInputTranscribed, UserStateChanged,
};
pub use crate::metrics::usage::AgentSessionUsage;
use lk_agents_host_derive::Host;

/// `id` and `created_at` are left unset, as in the reference (`remote_session.py:533`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallRef {
    pub name: String,
    pub arguments: String,
    pub call_id: String,
}

/// `id` and `created_at` are dropped by the reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCallOutputRef {
    pub call_id: String,
    pub name: String,
    pub output: String,
    pub is_error: bool,
}

/// `error` (tag 15). The whole error becomes one string.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "error")]
pub struct ErrorWire {
    pub message: String,
}

/// `overlapping_speech` (tag 16).
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "overlapping_speech")]
pub struct OverlappingSpeechWire {
    pub is_interruption: bool,
    /// Absent when the overlap has no recorded start, which the proto allows.
    #[binding(default)]
    pub overlap_started_at: Option<f64>,
    pub detection_delay: f64,
    pub detected_at: f64,
}

/// `amd_prediction` (tag 18). Back-channel: never passes through `session.on(..)`.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "amd_prediction")]
pub struct AmdPredictionWire {
    pub speech_duration: f64,
    /// The `(str, Enum)` value, read through `AmdCategory::from_wire`.
    pub category: AmdCategory,
    pub reason: String,
    pub transcript: String,
    pub delay: f64,
}

/// Tag 14. Read by hand below (`None` is an empty list).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionToolsExecutedWire {
    pub function_calls: Vec<FunctionCallRef>,
    pub function_call_outputs: Vec<FunctionCallOutputRef>,
}

/// Tag 20. Dead in the reference, but a peer may send it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionToolsStartedWire {
    pub function_calls: Vec<FunctionCallRef>,
}

#[cfg(feature = "python")]
mod tool_wires {
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    use super::*;

    fn calls(v: Option<Bound<'_, PyAny>>) -> PyResult<Vec<FunctionCallRef>> {
        let Some(v) = v.filter(|v| !v.is_none()) else { return Ok(Vec::new()) };
        Ok(v.extract::<Vec<(String, String, String)>>()?
            .into_iter()
            .map(|(name, arguments, call_id)| FunctionCallRef { name, arguments, call_id })
            .collect())
    }

    fn outputs(v: Option<Bound<'_, PyAny>>) -> PyResult<Vec<FunctionCallOutputRef>> {
        let Some(v) = v.filter(|v| !v.is_none()) else { return Ok(Vec::new()) };
        Ok(v.extract::<Vec<(String, String, String, bool)>>()?
            .into_iter()
            .map(|(call_id, name, output, is_error)| FunctionCallOutputRef { call_id, name, output, is_error })
            .collect())
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for FunctionToolsExecutedWire {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let d = obj.cast::<PyDict>()?;
            Ok(FunctionToolsExecutedWire {
                function_calls: calls(d.get_item("function_calls")?)?,
                function_call_outputs: outputs(d.get_item("function_call_outputs")?)?,
            })
        }
    }

    impl crate::python::HostReadTag for FunctionToolsExecutedWire {
        const KEY: &'static str = "kind";
        const TAG: &'static str = "function_tools_executed";
        const ITEMS: bool = true;
    }

    impl<'a, 'py> FromPyObject<'a, 'py> for FunctionToolsStartedWire {
        type Error = PyErr;

        fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
            let d = obj.cast::<PyDict>()?;
            Ok(FunctionToolsStartedWire { function_calls: calls(d.get_item("function_calls")?)? })
        }
    }

    impl crate::python::HostReadTag for FunctionToolsStartedWire {
        const KEY: &'static str = "kind";
        const TAG: &'static str = "function_tools_started";
        const ITEMS: bool = true;
    }
}

/// `debug_message` (tag 21): already a proto on both sides, so the serialized bytes cross.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "debug_message", default)]
pub struct DebugMessageWire {
    /// A serialized `livekit.agent.DebugMessage`, decoded by `remote_session::decode_debug_message`.
    pub payload: Option<Vec<u8>>,
}

// The `tool_execution_updated` sub-arms: a nested tagged union, the outer `kind` then the update's
// own `type`.

/// `tool_call_started`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "type", tag_value = "tool_call_started", default)]
pub struct ToolCallStartedWire {
    pub id: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

/// `session_usage_updated` (tag 17).
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "session_usage_updated", default)]
pub struct SessionUsageUpdatedWire {
    /// A list of dicts, each with its own `type`, read by the session report's reader.
    pub usage: AgentSessionUsage,
}

/// `tool_execution_updated` (tag 22): the update is its own tagged union, keyed by its `type`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract, from_items, tag = "kind", tag_value = "tool_execution_updated")]
pub struct ToolExecutionUpdatedWire {
    pub update: ToolUpdateWire,
}

/// One `tool_execution_updated` update, picked by its own `type`.
#[derive(Debug, Clone, PartialEq, Eq, Host)]
#[binding(extract)]
pub enum ToolUpdateWire {
    Started(ToolCallStartedWire),
    CallUpdated(ToolCallUpdated),
    Ended(ToolCallEnded),
    ReplyUpdated(ToolReplyUpdated),
}

/// One event `SessionHost` forwards, picked by its `kind`: what `_send_wire` hands the core.
#[derive(Debug, Clone, PartialEq, Host)]
#[binding(extract)]
pub enum WireEvent {
    AgentStateChanged(AgentStateChanged),
    UserStateChanged(UserStateChanged),
    UserInputTranscribed(UserInputTranscribed),
    Error(ErrorWire),
    OverlappingSpeech(OverlappingSpeechWire),
    EotPrediction(EotPrediction),
    AmdPrediction(AmdPredictionWire),
    FunctionToolsExecuted(FunctionToolsExecutedWire),
    FunctionToolsStarted(FunctionToolsStartedWire),
    DebugMessage(DebugMessageWire),
    ConversationItemAdded(ItemAddedWire),
    SessionUsageUpdated(SessionUsageUpdatedWire),
    ToolExecutionUpdated(ToolExecutionUpdatedWire),
    AgentFalseInterruption(AgentFalseInterruption),
}
