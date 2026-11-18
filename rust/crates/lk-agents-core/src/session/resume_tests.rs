use super::*;
use lk_agents_abi::session::{AgentInputRequest, InputRequestState};

#[test]
fn a_root_text_question_is_a_local_continuation_until_answered_or_closed() {
    for status in [
        InputRequestState::Open,
        InputRequestState::Responding,
        InputRequestState::Responded,
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
        let can_resume = matches!(status, InputRequestState::Responded | InputRequestState::Closed);
        assert_eq!(validate_voice(&state, "desk").is_ok(), can_resume);
    }
}
