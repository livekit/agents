use super::*;
use lk_agents_abi::session::{AgentInputRequest, InputRequestState};
use lk_agents_abi::{FunctionCall, FunctionCallOutput};
use std::time::UNIX_EPOCH;

#[test]
fn root_input_occurrences_do_not_restore_a_text_child_stack() {
    for status in [
        InputRequestState::Open,
        InputRequestState::Responding,
        InputRequestState::Responded,
        InputRequestState::Superseded,
        InputRequestState::Closed,
    ] {
        let mut state = SessionSnapshot::new("conversation".into(), "text".into(), "desk".into());
        state.frames.get_mut(&state.root_frame_id).unwrap().input_requests.insert(
            "date".into(),
            AgentInputRequest {
                state: status,
                source_message: lk_agents_abi::ChatMessage::text(
                    "date",
                    lk_agents_abi::Role::Assistant,
                    "Which date?",
                    std::time::UNIX_EPOCH,
                ),
                response_message: None,
                origin: None,
                presented_message_id: None,
            },
        );
        assert!(validate_voice(&state, "desk").is_ok());
    }
}

fn call(id: &str) -> ChatItem {
    FunctionCall {
        id: id.into(),
        call_id: "reused-provider-id".into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    }
    .into()
}

fn output(id: &str) -> ChatItem {
    FunctionCallOutput::new(id, "reused-provider-id", "done", false, UNIX_EPOCH).into()
}

#[test]
fn an_earlier_result_does_not_resolve_a_later_call_with_the_same_provider_id() {
    let history =
        [call("first-canonical-call"), output("opaque-first-result"), call("last-canonical-call")];
    assert!(validate_resume(&history).is_err());
}

#[test]
fn opaque_results_allow_sequential_provider_reuse_but_not_overlapping_ambiguity() {
    let history = [
        call("first-canonical-call"),
        output("opaque-first-result"),
        call("second-canonical-call"),
        output("opaque-second-result"),
    ];
    assert!(validate_resume(&history).is_ok());
    let overlapping =
        [history[0].clone(), history[2].clone(), history[1].clone(), history[3].clone()];
    assert!(validate_resume(&overlapping).is_err());
}

#[test]
fn a_progress_result_does_not_prove_the_live_local_body_finished() {
    let mut history = [call("provisional"), output("provisional_out")];
    let ChatItem::FunctionCall(call) = &mut history[0] else { unreachable!() };
    call.extra
        .insert(lk_agents_abi::llm::tool_executor::LIVE_LOCAL_PENDING_KEY.into(), true.into());
    assert!(validate_resume(&history).is_err());
    let ChatItem::FunctionCall(call) = &mut history[0] else { unreachable!() };
    call.extra.remove(lk_agents_abi::llm::tool_executor::LIVE_LOCAL_PENDING_KEY);
    assert!(validate_resume(&history).is_ok());
}
