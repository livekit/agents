//! Upstream reset codes, exported to the host without a second protocol definition.

#[crate::host(export(name = "HSR_ABORT", pid = false))]
pub const HSR_ABORT: i32 = livekit_protocol::agent_http::HttpStreamResetCode::HsrAbort as i32;

#[crate::host(export(name = "HSR_REFUSED", pid = false))]
pub const HSR_REFUSED: i32 = livekit_protocol::agent_http::HttpStreamResetCode::HsrRefused as i32;

#[crate::host(export(name = "HSR_INTERNAL", pid = false))]
pub const HSR_INTERNAL: i32 = livekit_protocol::agent_http::HttpStreamResetCode::HsrInternal as i32;

#[crate::host(export(name = "HSR_TIMEOUT", pid = false))]
pub const HSR_TIMEOUT: i32 = livekit_protocol::agent_http::HttpStreamResetCode::HsrTimeout as i32;

#[crate::host(export(name = "HSR_PROTOCOL", pid = false))]
pub const HSR_PROTOCOL: i32 = livekit_protocol::agent_http::HttpStreamResetCode::HsrProtocol as i32;
