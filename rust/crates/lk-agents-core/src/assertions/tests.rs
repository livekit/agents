use std::time::SystemTime;

use lk_agents_abi::ChatItem;
use lk_agents_abi::agent::assertions::{EventExpectation, RunEvents as _};
use lk_agents_abi::llm::chat_context::{ChatContent, Role};
use serde_json::json;

use super::RunEvents;

fn message(role: Role, text: &str) -> ChatItem {
    let content = vec![ChatContent::Text(text.into())];
    ChatItem::message("m".into(), role, content, SystemTime::UNIX_EPOCH, false, None, None, None)
}

fn call(name: &str, arguments: &str) -> ChatItem {
    let (time, args) = (SystemTime::UNIX_EPOCH, arguments.into());
    ChatItem::function_call("c".into(), "c1".into(), args, name.into(), time, None, None)
}

fn output(text: &str, is_error: bool) -> ChatItem {
    let time = SystemTime::UNIX_EPOCH;
    ChatItem::function_call_output("o".into(), "c1".into(), text.into(), is_error, time, None, true)
}

fn expect_call(name: &str, arguments: serde_json::Value) -> EventExpectation {
    EventExpectation::FunctionCall { name: Some(name.into()), arguments: Some(arguments) }
}

fn booking() -> RunEvents {
    RunEvents::new(vec![
        message(Role::User, "Book a room for tomorrow."),
        call("check_availability", r#"{"room_type": "king", "nights": 1.0, "late": false}"#),
        output("available", false),
        message(Role::Assistant, "A king room is available."),
    ])
}

#[test]
fn the_documented_sequence_walks_every_event_and_checks_supplied_arguments_only() {
    let mut events = booking();
    events.skip_next(1).unwrap();
    // Supplied keys only; `1` equals the stored `1.0`.
    let arguments = json!({"room_type": "king", "nights": 1});
    events.next_event(Some(expect_call("check_availability", arguments))).unwrap();
    let any_output = EventExpectation::FunctionCallOutput { output: None, is_error: None };
    events.next_event(Some(any_output)).unwrap();
    events.next_event(Some(EventExpectation::Message { role: Some("assistant".into()) })).unwrap();
    events.no_more_events().unwrap();
}

#[test]
fn a_mismatch_names_the_key_and_marks_the_event_in_its_context() {
    let events = booking();
    let failure =
        events.at(1, Some(expect_call("check_availability", json!({"room_type": "queen"}))));
    let message = failure.unwrap_err().0;
    assert!(
        message.starts_with(r#"For key 'room_type', expected "queen", got "king""#),
        "{message}"
    );
    assert!(message.contains(">>> [1] FunctionCallEvent(name='check_availability'"), "{message}");
    assert!(message.contains("    [0] ChatMessageEvent(role='user'"), "{message}");
}

#[test]
fn malformed_stored_arguments_fail_the_assertion_instead_of_raising_a_json_error() {
    let events = RunEvents::new(vec![call("lookup", "{not json")]);
    let message = events.at(0, Some(expect_call("lookup", json!({"id": 1})))).unwrap_err().0;
    assert!(
        message.starts_with("The call's arguments are not a JSON object: '{not json'"),
        "{message}"
    );
    assert!(message.contains(">>> [0] FunctionCallEvent"), "{message}");
}

#[test]
fn ranges_skips_and_the_end_of_events_report_their_own_failures() {
    let mut events = booking();
    let failed = EventExpectation::FunctionCallOutput { output: None, is_error: Some(true) };
    let message = events.contains(0, 4, failed).unwrap_err().0;
    assert!(
        message.starts_with("No FunctionCallOutputEvent satisfying criteria found in range [0:4]")
    );
    assert_eq!(events.contains(2, 99, EventExpectation::Message { role: None }).unwrap(), 3);

    // A non-matching conditional skip leaves the position; a matching one advances it.
    let assistant = EventExpectation::Message { role: Some("assistant".into()) };
    assert_eq!(events.skip_next_event_if(assistant), None);
    assert_eq!(events.skip_next_event_if(EventExpectation::Message { role: None }), Some(0));

    let message = events.skip_next(4).unwrap_err().0;
    assert!(message.starts_with("Tried to skip 4 event(s), but only 3 were available."));
    events.skip_next(3).unwrap();
    let message = events.next_event(None).unwrap_err().0;
    assert!(message.starts_with("Expected another event, but none left."));
    let message = events.at(-5, None).unwrap_err().0;
    assert!(message.starts_with("nth(-5) out of range (total events: 4)"));
}
