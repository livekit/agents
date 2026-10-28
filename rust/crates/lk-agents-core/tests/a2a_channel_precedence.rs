use lk_agents_abi::a2a_protocol as a2a;
use std::time::Duration;

use lk_agents_core::a2a::{
    DebounceSpec, Message, Policy, PolicySpec, channel_of, resolve_debounce, resolve_policy,
};
use lk_agents_abi::a2a::metadata::{CHANNEL_KEY, KIND_KEY};
use serde_json::json;

#[test]
fn channel_delegation_precedence_controls_context_and_inbox_defaults() {
    for (metadata, expected_channel, expected_policy, expected_debounce) in [
        (
            json!({CHANNEL_KEY: "sms", KIND_KEY: "delegation"}),
            "delegation",
            Policy::Queue,
            Duration::ZERO,
        ),
        (json!({CHANNEL_KEY: "sms"}), "sms", Policy::Batch, Duration::from_secs(2)),
        (json!({}), "a2a", Policy::Interrupt, Duration::ZERO),
    ] {
        let wire = serde_json::from_value::<a2a::Message>(json!({
            "messageId": "m1",
            "role": "ROLE_USER",
            "parts": [{"text": "help"}],
            "metadata": metadata,
        }))
        .unwrap();
        let channel = channel_of(&wire);
        let incoming = Message::from_a2a(&wire, 1.0, None).unwrap();
        assert_eq!(channel, expected_channel);
        assert_eq!(incoming.channel, channel, "the handler and inbox use the same resolver");
        let policy = resolve_policy(&PolicySpec::Default, &channel);
        assert_eq!(policy, expected_policy);
        assert_eq!(resolve_debounce(&DebounceSpec::Default, &channel, policy), expected_debounce);
    }
}
