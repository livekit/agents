//! Agent-card construction from the endpoint's descriptor, using upstream A2A wire records.

use std::collections::HashMap;

use a2a::{
    AgentCapabilities, AgentCard, AgentExtension, AgentInterface, AgentSkill,
    HttpAuthSecurityScheme, SecurityScheme,
};
use lk_agents_abi::a2a::card::{EndpointCard, EndpointDescriptor};
use lk_agents_abi::a2a::metadata::EXTENSION_URI;
use lk_agents_abi::errors::HostError;

pub struct CardBuilder;

pub fn agent_card(descriptor: EndpointDescriptor) -> Result<AgentCard, HostError> {
    if descriptor.name.is_empty() || descriptor.name.contains('/') {
        return Err(HostError::new("the endpoint name must be one path segment"));
    }
    if descriptor.description.trim().is_empty() || descriptor.version.trim().is_empty() {
        return Err(HostError::new("an endpoint needs a description and version"));
    }
    Ok(AgentCard {
        name: descriptor.name,
        description: descriptor.description.clone(),
        version: descriptor.version,
        supported_interfaces: vec![AgentInterface {
            url: descriptor.url,
            protocol_binding: a2a::TRANSPORT_PROTOCOL_JSONRPC.into(),
            protocol_version: a2a::VERSION.into(),
            tenant: None,
        }],
        capabilities: AgentCapabilities {
            streaming: Some(true),
            extensions: Some(vec![AgentExtension {
                uri: EXTENSION_URI.into(),
                description: None,
                required: Some(false),
                params: None,
            }]),
            ..Default::default()
        },
        default_input_modes: vec!["text/plain".into()],
        default_output_modes: vec!["text/plain".into()],
        skills: vec![AgentSkill {
            id: "delegate".into(),
            name: "delegate".into(),
            description: descriptor.description,
            tags: vec!["delegation".into()],
            examples: None,
            input_modes: None,
            output_modes: None,
            security_requirements: None,
        }],
        security_schemes: Some(HashMap::from([(
            "livekit".into(),
            SecurityScheme::HttpAuth(HttpAuthSecurityScheme {
                scheme: "bearer".into(),
                description: Some("A required LiveKit project token or thread grant".into()),
                bearer_format: Some("JWT".into()),
            }),
        )])),
        security_requirements: Some(vec![HashMap::from([("livekit".into(), vec![])])]),
        provider: None,
        documentation_url: None,
        icon_url: None,
        signatures: None,
    })
}

impl EndpointCard for CardBuilder {
    fn _core_a2a_agent_card(descriptor: EndpointDescriptor) -> Result<String, HostError> {
        serde_json::to_string(&agent_card(descriptor)?)
            .map_err(|error| HostError::new(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_publishes_the_exact_endpoint_descriptor_and_required_bearer() {
        let descriptor = EndpointDescriptor {
            name: "back-office".into(),
            description: "Books the itinerary and returns the reservation code.".into(),
            version: "deployment-42".into(),
            url: "https://example.test/back-office/v1".into(),
        };
        let card = agent_card(descriptor.clone()).unwrap();
        assert_eq!(card.name, descriptor.name);
        assert_eq!(card.version, descriptor.version);
        assert_eq!(card.skills.len(), 1);
        assert_eq!(card.skills[0].description, descriptor.description);
        assert_eq!(card.supported_interfaces.len(), 1);
        assert_eq!(card.supported_interfaces[0].url, descriptor.url);
        assert_eq!(card.supported_interfaces[0].protocol_binding, "JSONRPC");
        assert_eq!(
            card.security_requirements.unwrap(),
            vec![HashMap::from([("livekit".into(), vec![])])]
        );
    }
}
