//! The JSON-RPC door the worker serves an A2A endpoint through, and its already encoded
//! HTTP/SSE frames.

use std::sync::Arc;

use super::auth::AuthError;
use super::card::EndpointDescriptor;
use super::options::TurnOptions;
use super::turn::TurnHost;
use crate::errors::HostError;
use crate::storage::{StorageError, StorageNamespace};
use crate::worker::http::{EndpointExchange, EndpointRequest};

#[crate::host(record(module = "a2a_door", emit, extract))]
#[derive(Debug, Clone)]
pub struct DoorConfig {
    pub lane: String,
    pub holder: String,
    pub endpoint: EndpointDescriptor,
    pub options: TurnOptions,
}

/// One JSON-RPC response, as a JSON body and as an SSE event.
#[derive(Debug, Clone)]
pub struct DoorFrame {
    pub json: String,
    pub sse: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum DoorError {
    #[error(transparent)]
    Auth(#[from] AuthError),
    #[error(transparent)]
    Storage(#[from] StorageError),
}

/// What the worker serves one A2A request with, besides the exchange.
#[derive(Debug, Clone, Default)]
pub struct TextServing {
    /// The node's metadata, or the public listener's.
    pub request: EndpointRequest,
    /// The URL the card publishes: Cloud's, or the path the worker serves the endpoint at, put
    /// on the address the request reached. `None` before the server knows.
    pub endpoint_url: Option<String>,
    pub agent_name: String,
    pub deployment: String,
    /// The project key pair that verifies a bearer; `None` when the worker holds none.
    pub credentials: Option<(String, String)>,
    /// The host context the endpoint's handler runs in.
    pub host_context: Option<crate::HostRef>,
}

/// A native A2A door, as the worker serves it: its card at `AGENT_CARD_PATH` and its JSON-RPC
/// methods at `RPC_PATH`, under `/<endpoint>`.
#[crate::host(native(refused = "expected a native A2A door"))]
pub trait TextEndpoint: Send + Sync {
    /// The endpoint's path segment.
    fn name(&self) -> &str;

    /// Answer one exchange: authenticate the caller, then serve the card or one JSON-RPC call.
    fn serve(
        &self,
        exchange: Arc<dyn EndpointExchange>,
        serving: TextServing,
    ) -> crate::BoxFuture<'_, Result<(), HostError>>;
}

#[crate::host(class(name = "_core_A2ADoor", native = TextEndpoint))]
pub trait A2ADoor: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// The door over `storage`'s namespace, which opens its backend on first use; accepted work
    /// is settled by the storage's `aclose`.
    #[host(new)]
    fn new(
        env: &Self::Env,
        storage: Arc<dyn StorageNamespace>,
        config: DoorConfig,
        host: Arc<dyn TurnHost>,
    ) -> Result<Self, StorageError>;

    #[host(getter)]
    fn max_payload_bytes(&self) -> usize;

    #[host(await)]
    async fn aclose(&self);
}
