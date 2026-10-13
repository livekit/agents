//! `_connect_ws`'s exception mapping and the stream's failure errors, as pure functions. The retry
//! ladder is `lk_agents_abi::types::retry::decide(RequestKind::Interruption, ..)`.

use lk_agents_abi::errors::ApiError;

use crate::interruption::core::StreamError;

/// How the WebSocket handshake failed, at the granularity `_connect_ws` distinguishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectFailure {
    /// `asyncio.TimeoutError` from the `wait_for` around `ws_connect`.
    Timeout,
    /// `aiohttp.ClientResponseError` -- the server answered the upgrade with a status.
    Http { status: i64 },
    /// `aiohttp.ClientConnectorError` and everything else.
    Transport,
}

/// `_connect_ws`'s except block (`inference/interruption.py:1008`). A 429 and a handshake timeout
/// are not retryable; a plain connector error is.
pub fn connect_error(failure: ConnectFailure) -> ApiError {
    match failure {
        ConnectFailure::Http { status: 429 } => {
            ApiError::status("LiveKit Adaptive Interruption quota exceeded", 429, None, Some(false))
        }
        ConnectFailure::Timeout => ApiError::connection(Some(
            "failed to connect to LiveKit Adaptive Interruption: timeout".to_owned(),
        ))
        .with_retryable(false),
        ConnectFailure::Http { .. } | ConnectFailure::Transport => ApiError::connection(Some(
            "failed to connect to LiveKit Adaptive Interruption".to_owned(),
        )),
    }
}

/// `recv_task`'s unexpected-close branch (`inference/interruption.py:836`): `ws.close_code or -1`,
/// so a `0` code becomes `-1`.
pub fn closed_unexpectedly(close_code: Option<u16>, data: &str, extra: &str) -> StreamError {
    let status = match close_code {
        Some(0) | None => -1,
        Some(code) => code as i64,
    };
    StreamError {
        error: ApiError::status(
            format!("LiveKit Adaptive Interruption connection closed unexpectedly: {data}"),
            status,
            None,
            None,
        ),
        // `f"{ws_msg.data=} {ws_msg.extra=}"` uses `repr`; `{:?}` on a `&str` matches it.
        body: Some(format!("ws_msg.data={data:?} ws_msg.extra={extra:?}")),
    }
}

/// `_connect_ws`'s post-handshake failure (`inference/interruption.py:1030`): the socket opened but
/// `session.create` could not be written. Retryable, unlike the handshake failures above.
pub fn session_create_failed() -> ApiError {
    ApiError::connection(Some(
        "failed to send session.create message to LiveKit Adaptive Interruption".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a handshake timeout or a 429 marked retryable.
    #[test]
    fn a_handshake_timeout_is_not_retryable_but_a_connector_error_is() {
        assert!(!connect_error(ConnectFailure::Timeout).retryable);
        assert!(!connect_error(ConnectFailure::Http { status: 429 }).retryable);
        assert!(connect_error(ConnectFailure::Transport).retryable);
        // A non-429 HTTP answer takes the generic connection path, message and all.
        let other = connect_error(ConnectFailure::Http { status: 503 });
        assert_eq!(other.message, "failed to connect to LiveKit Adaptive Interruption");
        assert_eq!(other.status_code, None);
    }

    // Catches: `close_code == 0` passed through as a status (R2b).
    #[test]
    fn a_zero_or_absent_close_code_becomes_minus_one() {
        assert_eq!(closed_unexpectedly(Some(0), "", "").error.status_code, Some(-1));
        assert_eq!(closed_unexpectedly(None, "", "").error.status_code, Some(-1));
        assert_eq!(closed_unexpectedly(Some(1006), "", "").error.status_code, Some(1006));
    }
}
