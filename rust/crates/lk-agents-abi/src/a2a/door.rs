//! The JSON-RPC door, its request-scoped response and already encoded HTTP/SSE frames.

use std::sync::Arc;

use serde_json::Value;

use super::auth::{AuthError, AuthPrincipal};
use super::card::EndpointDescriptor;
use super::turn::TurnHost;
use crate::storage::StorageError;

#[crate::host(record(module = "a2a_door", emit, extract))]
#[derive(Debug, Clone)]
pub struct DoorConfig {
    pub lane: String,
    pub holder: String,
    pub endpoint: EndpointDescriptor,
    pub options: Value,
}

#[crate::host(record(module = "a2a_door", emit))]
#[derive(Debug, Clone)]
pub struct DoorFrame {
    pub json: String,
    pub sse: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
#[crate::host(error(with = crate::a2a::door::door_error_to_py))]
pub enum DoorError {
    #[error(transparent)]
    Auth(#[from] AuthError),
    #[error(transparent)]
    Storage(#[from] StorageError),
}

#[cfg(feature = "python")]
pub fn door_error_to_py(error: DoorError) -> pyo3::PyErr {
    match error {
        DoorError::Auth(error) => super::auth::auth_error_to_py(error),
        DoorError::Storage(error) => crate::storage::storage_error_to_py(error),
    }
}

#[crate::host(class(name = "_core_A2ADoor"))]
pub trait A2ADoor: Send + Sync + Sized + 'static {
    #[host(class(name = "_core_A2AResponse"))]
    type Response: Send + Sync + 'static;

    #[host(getter)]
    fn max_payload_bytes(&self) -> usize;

    #[host(await)]
    async fn dispatch(
        &self,
        body: Vec<u8>,
        principal: AuthPrincipal,
        #[host(kw, default = None)] version: Option<String>,
        #[host(default = None)] offset: Option<String>,
        #[host(default = None)] body_size: Option<usize>,
        #[host(default = None)] host_context: Option<crate::HostRef>,
    ) -> Self::Response;

    #[host(await)]
    async fn aclose(&self);
}

#[crate::host(class(name = "_core_A2AResponse"))]
pub trait A2AResponse: Send + Sync + Sized + 'static {
    #[host(getter)]
    fn streaming(&self) -> bool;

    #[host(await)]
    async fn next(&self) -> Result<Option<DoorFrame>, DoorError>;

    #[host(await)]
    async fn aclose(&self);
}

/// A factory parameter type shared with Storage's generated constructor method.
pub type Handler = Arc<dyn TurnHost>;
