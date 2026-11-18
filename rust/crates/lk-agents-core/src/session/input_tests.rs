use super::*;
use lk_agents_abi::session::{AgentInputRequest, ToolCallRef};
use std::time::UNIX_EPOCH;

fn state() -> SessionSnapshot {
    SessionSnapshot::new("conversation".into(), "execution".into(), "desk".into())
}

fn message(id: &str, role: Role, text: &str) -> ChatMessage {
    ChatMessage::text(id, role, text, UNIX_EPOCH)
}

fn reference(state: &SessionSnapshot, id: &str) -> InputRequestRef {
    InputRequestRef::Agent { agent_id: state.root_frame_id.value.clone(), request_id: id.into() }
}

fn registration(state: &SessionSnapshot, id: &str, call: Option<&str>) -> SessionInputCommand {
    SessionInputCommand::Retain {
        agent_id: state.root_frame_id.value.clone(),
        request_id: id.into(),
        input: Box::new(AgentInputRequest {
            state: InputRequestState::Open,
            source_message: message(id, Role::Assistant, "Which date?"),
            response_message: None,
            presented_message_id: None,
            origin: call.map(|call_id| ToolCallRef {
                agent_id: state.root_frame_id.value.clone(),
                call_id: call_id.into(),
            }),
        }),
    }
}

fn present(state: &mut SessionSnapshot, requests: Vec<InputRequestRef>) {
    state.chat_history.push(message("heard", Role::Assistant, "Which date works for you?"));
    apply(state, vec![SessionInputCommand::Presented { requests, message_id: "heard".into() }])
        .unwrap();
}

fn respond(state: &mut SessionSnapshot, requests: Vec<InputRequestRef>) {
    state.chat_history.push(message("human", Role::User, "Friday"));
    apply(state, vec![SessionInputCommand::Respond { requests, message_id: "human".into() }])
        .unwrap();
}

#[test]
fn delivery_consumes_only_the_exact_reserved_local_response_and_is_atomic() {
    let mut state = state();
    let first = reference(&state, "first");
    let second = reference(&state, "second");
    let commands = vec![
        registration(&state, "first", Some("call-first")),
        registration(&state, "second", Some("call-second")),
    ];
    apply(&mut state, commands).unwrap();
    present(&mut state, vec![first.clone(), second.clone()]);
    respond(&mut state, vec![first.clone()]);
    let reserved = state.clone();
    assert!(
        apply(
            &mut state,
            vec![SessionInputCommand::Delivered {
                requests: vec![first.clone(), second],
                message_id: "human".into(),
            }]
        )
        .is_err()
    );
    assert_eq!(state, reserved, "an unreserved sibling cannot partly consume a group");

    for _ in 0..2 {
        apply(
            &mut state,
            vec![SessionInputCommand::Delivered {
                requests: vec![first.clone()],
                message_id: "human".into(),
            }],
        )
        .unwrap();
    }
    let inputs = &state.frames[&state.root_frame_id].input_requests;
    assert_eq!(inputs["first"].state, InputRequestState::Responded);
    assert_eq!(inputs["second"].state, InputRequestState::Open);
}

#[test]
fn registration_never_aliases_a_live_call_or_reopens_an_answered_occurrence() {
    let mut state = state();
    let first = registration(&state, "first", Some("call"));
    apply(&mut state, vec![first.clone()]).unwrap();
    let before = state.clone();
    let alias = registration(&state, "alias", Some("call"));
    assert!(apply(&mut state, vec![alias]).is_err());
    let changed_origin = registration(&state, "first", Some("different-call"));
    assert!(apply(&mut state, vec![changed_origin]).is_err());
    assert_eq!(state, before);

    let request = reference(&state, "first");
    present(&mut state, vec![request.clone()]);
    respond(&mut state, vec![request.clone()]);
    apply(
        &mut state,
        vec![SessionInputCommand::Delivered {
            requests: vec![request],
            message_id: "human".into(),
        }],
    )
    .unwrap();
    let answered = state.clone();
    apply(&mut state, vec![first]).unwrap();
    assert_eq!(state, answered, "a replayed retain preserves the response lifecycle");
    let next = registration(&state, "next-question", Some("call"));
    apply(&mut state, vec![next]).unwrap();
    assert_eq!(
        state.frames[&state.root_frame_id].input_requests["next-question"].state,
        InputRequestState::Open
    );
}

#[test]
fn closing_a_dead_bodys_occurrence_preserves_other_local_questions() {
    let mut state = state();
    let body = reference(&state, "body");
    let sibling = reference(&state, "sibling");
    let commands = vec![
        registration(&state, "body", Some("call-body")),
        registration(&state, "sibling", Some("call-sibling")),
        registration(&state, "generic", None),
    ];
    apply(&mut state, commands).unwrap();
    present(&mut state, vec![body.clone(), sibling.clone()]);
    respond(&mut state, vec![body.clone()]);
    apply(&mut state, vec![SessionInputCommand::Close { requests: vec![body] }]).unwrap();
    let inputs = &state.frames[&state.root_frame_id].input_requests;
    assert_eq!(inputs["body"].state, InputRequestState::Closed);
    assert_eq!(inputs["sibling"].state, InputRequestState::Open);
    assert_eq!(inputs["generic"].state, InputRequestState::Open);
    let before = state.clone();
    let unknown = reference(&state, "unknown");
    assert!(
        apply(&mut state, vec![SessionInputCommand::Close { requests: vec![sibling, unknown] }])
            .is_err()
    );
    assert_eq!(state, before);
}

#[test]
fn a_same_id_transcript_revision_requires_correction_before_local_delivery() {
    let mut state = state();
    let request = reference(&state, "first");
    let retain = registration(&state, "first", Some("call"));
    apply(&mut state, vec![retain]).unwrap();
    present(&mut state, vec![request.clone()]);
    respond(&mut state, vec![request.clone()]);
    *state.chat_history.iter_mut().find(|message| message.id == "human").unwrap() =
        message("human", Role::User, "Actually Saturday");
    let revised = state.clone();
    assert!(
        apply(
            &mut state,
            vec![SessionInputCommand::Delivered {
                requests: vec![request.clone()],
                message_id: "human".into(),
            }]
        )
        .is_err()
    );
    assert_eq!(state, revised);
    apply(
        &mut state,
        vec![SessionInputCommand::Correct {
            request: request.clone(),
            previous_message_id: "human".into(),
            message_id: "human".into(),
        }],
    )
    .unwrap();
    apply(
        &mut state,
        vec![SessionInputCommand::Delivered {
            requests: vec![request],
            message_id: "human".into(),
        }],
    )
    .unwrap();
    let input = &state.frames[&state.root_frame_id].input_requests["first"];
    assert_eq!(input.state, InputRequestState::Responded);
    assert_eq!(
        input.response_message.as_ref().unwrap().raw_text_content().as_deref(),
        Some("Actually Saturday")
    );
}
