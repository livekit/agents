//! The endpoint's agent card (docs/protocol/a2a.md#the-agent-card), from its descriptor, as the
//! upstream A2A card's JSON in its field order.

use lk_agents_abi::a2a::card::{EndpointCard, EndpointDescriptor};
use lk_agents_abi::a2a::metadata::EXTENSION_URI;
use lk_agents_abi::a2a_protocol as a2a;
use lk_agents_abi::errors::HostError;
use serde_json::{Value, json};

pub struct CardBuilder;

pub fn agent_card(descriptor: EndpointDescriptor) -> Result<Value, HostError> {
    if descriptor.name.is_empty() || descriptor.name.contains('/') {
        return Err(HostError::new("the endpoint name must be one path segment"));
    }
    if descriptor.description.trim().is_empty() || descriptor.version.trim().is_empty() {
        return Err(HostError::new("an endpoint needs a description and version"));
    }
    Ok(json!({
        "name": descriptor.name,
        "description": descriptor.description,
        "version": descriptor.version,
        "supportedInterfaces": [{
            "url": descriptor.url,
            "protocolBinding": a2a::TRANSPORT_PROTOCOL_JSONRPC,
            "protocolVersion": a2a::VERSION,
        }],
        "capabilities": {
            "streaming": true,
            "extensions": [{"uri": EXTENSION_URI, "required": false}],
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [{
            "id": "delegate",
            "name": "delegate",
            "description": descriptor.description,
            "tags": ["delegation"],
        }],
        "securitySchemes": {"livekit": {"httpAuthSecurityScheme": {
            "scheme": "bearer",
            "description": "A required LiveKit project token or thread grant",
            "bearerFormat": "JWT",
        }}},
        "securityRequirements": [{"schemes": {"livekit": {"list": []}}}],
    }))
}

impl EndpointCard for CardBuilder {
    fn _core_a2a_extensions_header(requested: Option<String>) -> Option<String> {
        crate::a2a::rpc::extensions_header(requested.as_deref())
    }

    fn _core_a2a_agent_card(descriptor: EndpointDescriptor) -> Result<String, HostError> {
        Ok(agent_card(descriptor)?.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: the card drifting from the upstream record (a client parses it as one) or from
    // the descriptor and the required bearer.
    #[test]
    fn card_publishes_the_exact_endpoint_descriptor_and_required_bearer() {
        let descriptor = EndpointDescriptor {
            name: "back-office".into(),
            description: "Books the itinerary and returns the reservation code.".into(),
            version: "deployment-42".into(),
            url: "https://example.test/back-office/v1".into(),
        };
        let json = agent_card(descriptor.clone()).unwrap();
        let card: a2a::AgentCard = serde_json::from_value(json.clone()).expect("an agent card");
        assert_eq!(serde_json::to_value(&card).unwrap(), json, "the upstream record's wire form");
        assert_eq!((card.name, card.version), (descriptor.name, descriptor.version));
        assert_eq!(card.skills[0].description, descriptor.description);
        assert_eq!(card.supported_interfaces[0].url, descriptor.url);
        assert_eq!(card.supported_interfaces[0].protocol_binding, "JSONRPC");
        assert!(card.security_schemes.unwrap().contains_key("livekit"));
    }
}
