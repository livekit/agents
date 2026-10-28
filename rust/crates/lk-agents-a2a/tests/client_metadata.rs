//! The generated client keeps application metadata distinct from framework wire keys.

use lk_agents_a2a::client::{message_from_wire, message_to_wire};
use lk_agents_abi::a2a::client::{MessageInput, MessageMetadata};
use lk_agents_storage::run::EXTENSION_URI;
use serde_json::json;

#[test]
fn short_livekit_metadata_survives_send_without_framework_semantics() {
    let application = json!({
        "livekit/kind": "delegation",
        "livekit/channel": "sms",
        "livekit/verbatim": "application text",
        "livekit/answerTool": "application tool",
        "livekit/mergedInto": "application task",
        "livekit/reason": {"application": "reason"},
        "livekit/interrupted": "application value",
        "livekit/interruptionReason": ["application reason"],
    })
    .as_object()
    .unwrap()
    .clone();
    let mut input = message_from_wire(
        serde_json::from_value(json!({
            "messageId": "message", "role": "ROLE_USER", "parts": [{"text": "question"}],
        }))
        .unwrap(),
    );
    input.metadata = Some(MessageMetadata {
        channel: Some("chat".into()),
        application: Some(application.clone()),
        ..Default::default()
    });
    let wire = message_to_wire(MessageInput::Message(input), "generated-id".into()).unwrap();
    let received = lk_agents_a2a::Message::from_a2a(&wire, 0.0, None).unwrap();
    assert_eq!(received.channel, "chat");
    let mut decoded = message_from_wire(wire);
    let metadata = decoded.metadata.take().unwrap();
    assert_eq!(metadata.kind, None);
    assert_eq!(metadata.channel.as_deref(), Some("chat"));
    assert_eq!(metadata.application, Some(application));

    let mut colliding = decoded;
    colliding.metadata = Some(MessageMetadata {
        application: Some(
            [(format!("{EXTENSION_URI}/channel"), json!("spoofed"))].into_iter().collect(),
        ),
        ..Default::default()
    });
    assert!(message_to_wire(MessageInput::Message(colliding), "other-id".into()).is_err());
}
