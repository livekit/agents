//! GENERATED FILE -- DO NOT EDIT. Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`); see `napi.rs` in `lk-agents-codegen`.

#![allow(clippy::all, dead_code, non_snake_case, unused_imports, unused_variables, unused_mut)]

use crate::host_error::JsError;

/// `StrictError` as a host error (`#[host(error)]` in `rust/crates/lk-agents-abi/src/llm/provider_format/strict.rs`).
#[allow(unreachable_patterns)]
pub fn strict_error_to_node(e: lk_agents_abi::llm::provider_format::StrictError) -> JsError {
    let kind = match &e {
        lk_agents_abi::llm::provider_format::StrictError::NotAnObject { .. } => "TypeError",
        lk_agents_abi::llm::provider_format::StrictError::RefUnresolvable { .. } => "KeyError",
        _ => "ValueError",
    };
    JsError::new(kind, e.to_string())
}

/// `McpError` as a host error (`#[host(error)]` in `rust/crates/lk-agents-inference/src/mcp/mod.rs`).
#[allow(unreachable_patterns)]
pub fn mcp_error_to_node(e: lk_agents_inference::mcp::McpError) -> JsError {
    let kind = match &e {
        lk_agents_inference::mcp::McpError::Unavailable => "livekit.agents.llm.tool_context.ToolError",
        lk_agents_inference::mcp::McpError::EmptyResult(..) => "livekit.agents.llm.tool_context.ToolError",
        lk_agents_inference::mcp::McpError::ToolFailed(..) => "livekit.agents.llm.tool_context.ToolError",
        lk_agents_inference::mcp::McpError::InvalidTransport(..) => "ValueError",
        _ => "RuntimeError",
    };
    JsError::new(kind, e.to_string())
}

/// `TaskError` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/workflows/handle.rs`).
pub fn task_error_to_node(e: lk_agents_voice::workflows::TaskError) -> JsError {
    JsError::new("RuntimeError", e.to_string())
}

/// `SpeechRefused` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/speech_handle.rs`).
pub fn speech_refused_to_node(e: lk_agents_voice::speech_handle::SpeechRefused) -> JsError {
    JsError::new("RuntimeError", e.to_string())
}

/// `OpenRefused` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/agent_session/api.rs`).
pub fn open_refused_to_node(e: lk_agents_voice::agent_session::OpenRefused) -> JsError {
    JsError::new("RuntimeError", e.to_string())
}

/// `StartFailed` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/agent_session/api.rs`).
pub fn start_failed_to_node(e: lk_agents_voice::agent_session::StartFailed) -> JsError {
    JsError::new("RuntimeError", e.to_string())
}

/// `SessionRefused` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/agent_session/api.rs`).
pub fn session_refused_to_node(e: lk_agents_voice::agent_session::SessionRefused) -> JsError {
    JsError::new("RuntimeError", e.to_string())
}

/// `UnknownCloseReason` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/agent_session/api.rs`).
pub fn unknown_close_reason_to_node(e: lk_agents_voice::agent_session::UnknownCloseReason) -> JsError {
    JsError::new("ValueError", e.to_string())
}

/// `CommandRefused` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/agent_session/api.rs`).
#[allow(unreachable_patterns)]
pub fn command_refused_to_node(e: lk_agents_voice::agent_session::CommandRefused) -> JsError {
    let kind = match &e {
        lk_agents_voice::agent_session::CommandRefused::Session(..) => "RuntimeError",
        _ => "ValueError",
    };
    JsError::new(kind, e.to_string())
}

/// `NotInterruptible` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/agent_session/api.rs`).
pub fn not_interruptible_to_node(e: lk_agents_voice::agent_session::NotInterruptible) -> JsError {
    JsError::new("RuntimeError", e.to_string())
}

/// `RequestError` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/remote_session/driver.rs`).
#[allow(unreachable_patterns)]
pub fn request_error_to_node(e: lk_agents_voice::remote_session::driver::RequestError) -> JsError {
    let kind = match &e {
        lk_agents_voice::remote_session::driver::RequestError::Transport(inner) => "=HostError",
        lk_agents_voice::remote_session::driver::RequestError::Timeout => "TimeoutError",
        lk_agents_voice::remote_session::driver::RequestError::Cancelled => "asyncio.CancelledError",
        lk_agents_voice::remote_session::driver::RequestError::Malformed(..) => "ValueError",
        _ => "RuntimeError",
    };
    JsError::new(kind, e.to_string())
}

/// `ReadyError` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/remote_session/driver.rs`).
#[allow(unreachable_patterns)]
pub fn ready_error_to_node(e: lk_agents_voice::remote_session::driver::ReadyError) -> JsError {
    let kind = match &e {
        lk_agents_voice::remote_session::driver::ReadyError::Timeout => "TimeoutError",
        lk_agents_voice::remote_session::driver::ReadyError::Transport(inner) => "=HostError",
        lk_agents_voice::remote_session::driver::ReadyError::Cancelled => "asyncio.CancelledError",
        _ => "RuntimeError",
    };
    JsError::new(kind, e.to_string())
}

/// `IoError` as a host error (`#[host(error)]` in `rust/crates/lk-agents-abi/src/voice/io.rs`).
pub fn io_error_to_node(e: lk_agents_abi::voice::io::IoError) -> JsError {
    JsError::new("RuntimeError", e.to_string())
}

/// `RoomRefused` as a host error (`#[host(error)]` in `rust/crates/lk-agents-voice/src/room_io/rooms.rs`).
pub fn room_refused_to_node(e: lk_agents_voice::room_io::rooms::RoomRefused) -> JsError {
    JsError::new("ValueError", e.to_string())
}
