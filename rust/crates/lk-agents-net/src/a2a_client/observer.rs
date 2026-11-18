//! The conversation owner records intent and observations before dependent network work.

use lk_agents_abi::BoxFuture;
use lk_agents_core::a2a::client::{ClientError, Request};

pub trait RequestObserver: Send + Sync {
    fn prepared<'a>(&'a self, request: &'a Request) -> BoxFuture<'a, Result<(), ClientError>>;
    fn dispatch_started(&self) -> BoxFuture<'_, Result<(), ClientError>>;
    /// Checkpoint the raw SDK value before returning the owner's accepted observation.
    /// This prevents a stale poll from reopening an already accepted terminal task.
    fn response<'a>(
        &'a self,
        request: &'a Request,
        response: a2a::StreamResponse,
    ) -> BoxFuture<'a, Result<a2a::StreamResponse, ClientError>>;
    fn ended(&self, error: Option<ClientError>) -> BoxFuture<'_, Result<(), ClientError>>;
}
