//! A byte-frame duplex, one whole message per `Vec<u8>`. Two objects, because every protocol sends
//! and receives concurrently.

use async_trait::async_trait;

use crate::errors::ApiError;

#[async_trait]
pub trait WebSocketSink: Send {
    async fn send(&mut self, data: Vec<u8>) -> Result<(), ApiError>;
}

/// The read half. `None` means the peer closed; `Some(Err(..))` is a transport-level failure.
#[async_trait]
pub trait WebSocketStream: Send {
    async fn recv(&mut self) -> Option<Result<Vec<u8>, ApiError>>;
}
