//! Incoming SDK streams use the shared pull/close primitive; runtime owns admission.

use super::{ParticipantIdentity, RtcError, StreamInfo};
use crate::{Closed, StreamSource, Ts};
use std::sync::Arc;

pub type Receiver<T> = Box<dyn StreamSource<T>>;

pub const MAX_INCOMING_STREAMS: usize = 4;
pub const MAX_STREAM_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_STREAM_CHUNK_BYTES: usize = 64 * 1024;

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
    pub async fn read_all(&mut self) -> Result<String, RtcError> {
        let mut text = String::new();
        while let Some(chunk) = self.chunks.next().await? {
            if chunk.len() > MAX_STREAM_BYTES.saturating_sub(text.len()) {
                return Err(RtcError::StreamLimit {
                    resource: "text bytes",
                    limit: MAX_STREAM_BYTES,
                });
            }
            text.push_str(&chunk);
        }
        Ok(text)
    }

    pub async fn close(self, deadline: Ts) -> Closed {
        self.chunks.close(deadline).await
    }
}

/// The callback's reader is accepted before any asynchronous pull or payload copy.
pub trait ByteStreamIngress: Send + Sync {
    fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Receiver<Vec<u8>>,
    ) -> bool;
}

#[crate::class(from_py_object)]
#[derive(Clone)]
pub struct ByteStreamSender(pub Arc<dyn ByteStreamIngress>);

#[crate::host(object)]
impl ByteStreamSender {
    #[host(method)]
    pub fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Box<dyn StreamSource<Vec<u8>>>,
    ) -> bool {
        self.0.open(info, sender, chunks)
    }
}

pub trait TextStreamIngress: Send + Sync {
    fn open(&self, info: StreamInfo, sender: ParticipantIdentity, chunks: Receiver<String>)
    -> bool;
}

#[crate::class(from_py_object)]
#[derive(Clone)]
pub struct TextStreamSender(pub Arc<dyn TextStreamIngress>);

#[crate::host(object)]
impl TextStreamSender {
    #[host(method)]
    pub fn open(
        &self,
        info: StreamInfo,
        sender: ParticipantIdentity,
        chunks: Box<dyn StreamSource<String>>,
    ) -> bool {
        self.0.open(info, sender, chunks)
    }
}

/// The SDK's registered RPC callback.
#[crate::class(from_py_object)]
#[derive(Clone)]
pub struct RpcHandler(pub Arc<dyn Fn(&str, &str) -> String + Send + Sync>);

impl RpcHandler {
    pub fn new(callback: impl Fn(&str, &str) -> String + Send + Sync + 'static) -> Self {
        Self(Arc::new(callback))
    }
}

#[crate::host(object)]
impl RpcHandler {
    #[host(method)]
    pub fn call(&self, caller_identity: String, payload: String) -> String {
        (self.0)(&caller_identity, &payload)
    }
}
