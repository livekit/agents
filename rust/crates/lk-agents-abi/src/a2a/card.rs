//! The endpoint identity published through its agent card.

use crate::errors::HostError;

#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "a2a_card", emit, extract))]
pub struct EndpointDescriptor {
    pub name: String,
    pub description: String,
    pub version: String,
    pub url: String,
}

#[crate::host(functions)]
pub trait EndpointCard: Send + Sync + Sized + 'static {
    /// Serialize the upstream A2A card; the host only adapts it to its HTTP/SDK representation.
    #[host(function)]
    fn _core_a2a_agent_card(descriptor: EndpointDescriptor) -> Result<String, HostError>;
}
