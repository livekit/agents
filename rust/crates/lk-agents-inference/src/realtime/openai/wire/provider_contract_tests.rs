use super::*;
use crate::realtime::{OPENAI, protocol::Incoming, testfix};
use lk_agents_abi::Ts;

/// Recorded synthetic requests against gpt-realtime-mini. In particular, runtime validation
/// may omit the client's event_id, and generation failure is not a top-level error envelope.
#[test]
fn observed_openai_failures_preserve_correlation_and_reach_the_machine() {
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("fixtures/openai_errors.json")).unwrap();
    assert!(fixtures.len() >= 30);
    for fixture in fixtures {
        let name = fixture["case"].as_str().unwrap();
        let event = &fixture["event"];
        let incoming =
            decode_text(&event.to_string(), &WireShape::default(), testfix::TIME).unwrap();
        let mut machine = testfix::machine(OPENAI);
        machine.open(Ts::ZERO);
        match incoming {
            Incoming::Error(ref error) => {
                assert_eq!(error.code.as_deref(), event["error"]["code"].as_str(), "{name}");
                assert_eq!(
                    error.event_id.as_deref(),
                    event["error"]["event_id"].as_str(),
                    "{name}"
                );
                let benign = matches!(
                    error.code.as_deref(),
                    Some("response_cancel_not_active" | "input_audio_buffer_commit_empty")
                );
                let step = machine.incoming_on(machine.connection_generation(), Ts::ZERO, incoming);
                assert!(step.fatal.is_none(), "{name}: validation errors leave the session usable");
                assert_eq!(
                    step.events
                        .iter()
                        .any(|e| matches!(e, crate::realtime::Event::Error(e) if e.recoverable)),
                    !benign,
                    "{name}"
                );
                assert!(!machine.has_active_generation(), "{name}");
            }
            Incoming::ResponseDone(done) => {
                assert_eq!(name, "response_invalid_tool_choice");
                assert_eq!(done.status, lk_agents_abi::llm::realtime::GenerationStatus::Failed);
                assert_eq!(done.error_code.as_deref(), Some("invalid_tool_choice"));
                assert_eq!(done.error_type.as_deref(), Some("invalid_request_error"));
            }
            other => panic!("{name}: unexpected {other:?}"),
        }
    }
}

#[test]
fn an_error_message_alone_does_not_turn_a_real_failure_into_a_cancel_race() {
    let mut machine = testfix::machine(OPENAI);
    machine.open(Ts::ZERO);
    let step = machine.incoming_on(
        machine.connection_generation(),
        Ts::ZERO,
        Incoming::Error(crate::realtime::protocol::ServerError {
            event_id: None,
            type_: Some("invalid_request_error".into()),
            code: Some("response_cancel_not_found".into()),
            message: "Cancellation failed: response does not exist".into(),
        }),
    );
    assert!(step.events.iter().any(|e| matches!(e, crate::realtime::Event::Error(e) if e.code.as_deref() == Some("response_cancel_not_found"))));
}
