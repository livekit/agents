use super::*;
use crate::agent_session::Now;
use crate::agent_session::tests::realtime_driver;
use crate::generation::GenEvent;
use lk_agents_abi::voice::events::AgentState;

/// When a realtime model never sends the reply its tools asked for, the agent listens again.
/// https://github.com/livekit/agents/pull/5725 https://github.com/livekit/agents/issues/1155
#[test]
#[ignore = "https://github.com/livekit/agents/pull/5725 reproduces: auto_reply_expected survives the follow-up timeout, so the agent stays thinking"]
fn a_missing_auto_tool_reply_returns_the_agent_to_listening() {
    let mut driver = realtime_driver();
    let rt = driver.activity.realtime_activity_mut().expect("a realtime activity");
    rt.state.caps.auto_tool_reply_generation = true;
    rt.session.capabilities.auto_tool_reply_generation = true;
    let request =
        Request::GenerateReply { instructions: None, tool_choice: None, read_back: false };
    let speech = driver
        .realtime_request(Ts::ZERO, request, None, None, VoiceRunParticipation::Outside)
        .expect("a speech for the reply");
    driver.settle_step();
    let created =
        rt::GenerationCreated { response_id: "resp-1".into(), req_id: Some("_rt_0002".into()) };
    driver.on_realtime(Ts::ZERO, RealtimeWake::Event(Event::GenerationCreated(created)));
    driver.settle_step();
    let wake = RealtimeWake::ToolsDone { speech, step: 1, ask: true, tool_choice: None };
    driver.on_realtime(Ts::ZERO, wake);
    // Its tools ran: the agent thinks while the server composes the reply they asked for.
    driver.set_agent_state(Ts::ZERO, AgentState::Thinking);
    driver.settle_step();
    driver.take_outbox();

    // The server never starts the follow-up: its 5 s wait runs out.
    let late = Ts::ZERO.saturating_add(AUTO_TOOL_REPLY_TIMEOUT);
    let effects = driver.handle(None, Now { at: late, wall: std::time::UNIX_EPOCH });
    assert!(
        effects.iter().any(|effect| matches!(effect,
            crate::agent_session::effects::VoiceEffect::CloseRealtimePlayer(id) if *id == speech)),
        "the speech ends on what it spoke"
    );
    let outcome = crate::generation::TurnOutcome {
        forwarded_text: "Let me check.".into(),
        assistant_message: None,
        cut_at: None,
    };
    let done = GenEvent::Done { speech, outcome: Box::new(outcome), interrupted: false };
    driver.on_generation(late, done);
    assert_eq!(
        driver.session.states.agent_state(),
        AgentState::Listening,
        "a timed-out follow-up still holds the agent thinking"
    );
}
