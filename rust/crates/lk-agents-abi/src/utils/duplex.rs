//! A byte-frame duplex, one whole message per `Vec<u8>`. Two objects, because every protocol sends
//! and receives concurrently.

use crate::errors::ApiError;

pub trait WebSocketSink: Send {
    fn send(&mut self, data: Vec<u8>) -> crate::BoxFuture<'_, Result<(), ApiError>>;
}

/// The read half. `None` means the peer closed; `Some(Err(..))` is a transport-level failure.
pub trait WebSocketStream: Send {
    fn recv(&mut self) -> crate::BoxFuture<'_, Option<Result<Vec<u8>, ApiError>>>;
}
