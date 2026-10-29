//! Incoming SDK streams use the shared pull/close primitive; runtime admits them within voice's
//! bounds.

use super::{ParticipantIdentity, StreamInfo};
use crate::{Closed, StreamSource, Ts};
use std::any::Any;

pub type Receiver<T> = Box<dyn StreamSource<T>>;

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
        chunks: Box<dyn StreamSource<Vec<u8>>>,
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
        chunks: Box<dyn StreamSource<String>>,
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
