//! `_connect_ws`'s exception mapping and the stream's failure errors, as pure functions. The retry
//! allowance and backoff come from `lk_agents_core::attempts::RequestPolicy`.

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

/// `recv_task`'s unexpected-close branch (`inference/interruption.py:836`), `status` the
/// [`close_status`](lk_agents_net::transport::close_status).
pub fn closed_unexpectedly(status: i64, data: &str, extra: &str) -> StreamError {
    StreamError {
        request: None,
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
}
