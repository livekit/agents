//! Incoming SDK streams use the shared pull/close primitive; runtime admits them within voice bounds.

use super::{ParticipantIdentity, StreamInfo};
use crate::{Closed, StreamOutput, Ts};
use std::any::Any;
use std::sync::Arc;

pub type Receiver<T> = Box<dyn StreamOutput<T>>;

pub struct ByteStream {
    pub info: StreamInfo,
    pub sender: ParticipantIdentity,
    pub chunks: Receiver<Vec<u8>>,
}

impl ByteStream {
    pub async fn close(self, deadline: Ts) -> Closed {
        self.chunks.close(deadline).await
    }
}

pub struct TextStream {
    pub info: StreamInfo,
    pub sender: ParticipantIdentity,
    pub chunks: Receiver<String>,
}

impl TextStream {
    pub async fn close(self, deadline: Ts) -> Closed {
        self.chunks.close(deadline).await
    }
}

/// Where the host hands each byte stream opened on a topic; the reader is accepted before any
/// asynchronous pull or payload copy.
#[crate::host(native(refused = "expected a _core.ByteStreamSender"))]
pub trait ByteStreamIngress: Any + Send + Sync {
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Receiver<Vec<u8>>,
    ) -> bool;
}

#[crate::host(class(name = "ByteStreamSender", native = ByteStreamIngress))]
pub trait ByteStreamSender: Send + Sync + Sized + 'static {
    #[host(call)]
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Box<dyn StreamOutput<Vec<u8>>>,
    ) -> bool;
}

/// [`ByteStreamIngress`] for text streams.
#[crate::host(native(refused = "expected a _core.TextStreamSender"))]
pub trait TextStreamIngress: Any + Send + Sync {
    fn open(&self, info: StreamInfo, sender: ParticipantIdentity, chunks: Receiver<String>)
    -> bool;
}

#[crate::host(class(name = "TextStreamSender", native = TextStreamIngress))]
pub trait TextStreamSender: Send + Sync + Sized + 'static {
    #[host(call)]
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Box<dyn StreamOutput<String>>,
    ) -> bool;
}

/// The SDK's registered RPC callback: `(caller_identity, payload)` to the response payload.
#[crate::host(native(refused = "expected a _core.RpcHandler"))]
pub trait RpcCallback: Any + Send + Sync {
    fn call(&self, caller_identity: &str, payload: &str) -> String;
}

#[crate::host(class(name = "RpcHandler", native = RpcCallback))]
pub trait RpcHandler: Send + Sync + Sized + 'static {
    #[host(call, name = "call")]
    fn respond(&self, caller_identity: String, payload: String) -> String;
}

/// Routes shared by receivers attached to one SDK room instance. Runtime owns registrations.
#[crate::host(native(refused = "expected the room's native RPC router"))]
pub trait RpcRoutes: Send + Sync {
    fn insert(&self, caller: String, callback: Arc<dyn RpcCallback>) -> u64;
    fn remove(&self, caller: &str, registration: u64);
    fn call(&self, caller: &str, payload: &str) -> String;
}

/// Private room-adapter service; unrelated rooms never share mutable RPC routes.
#[crate::host(class(name = "_core_RoomRpcRouter", native = RpcRoutes))]
pub trait RoomRpcRouter: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new() -> Self;
}
