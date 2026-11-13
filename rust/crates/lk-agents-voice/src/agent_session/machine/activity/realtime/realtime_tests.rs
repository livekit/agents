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

/// A reply asked of a session that is swapped out before answering keeps its speech and is asked
/// of the new session, with its instructions, after the new session's frame.
#[test]
fn a_reply_the_old_session_never_answered_is_asked_again_after_a_swap() {
    use crate::agent_session::effects::VoiceEffect;
    let mut driver = realtime_driver();
    let request = Request::GenerateReply {
        instructions: Some("greet the user".into()),
        tool_choice: None,
        read_back: false,
    };
    let speech = driver
        .realtime_request(Ts::ZERO, request, None, None, VoiceRunParticipation::Outside)
        .expect("a speech for the reply");
    let asked = |effects: &[VoiceEffect]| {
        effects.iter().position(|e| {
            matches!(e, VoiceEffect::Realtime { command: RealtimeCommand::GenerateReply(r), .. }
                if r.instructions.as_deref() == Some("greet the user"))
        })
    };
    assert!(asked(&driver.settle_step()).is_some(), "the reply is asked of the first session");

    driver.on_realtime(Ts::ZERO, RealtimeWake::Event(Event::SessionSwapped { regenerate: true }));
    let effects = driver.settle_step();
    let frame = effects.iter().position(|e| {
        matches!(e, VoiceEffect::Realtime { command: RealtimeCommand::UpdateChatCtx(_), .. })
    });
    let replies = effects
        .iter()
        .filter(|e| {
            matches!(e, VoiceEffect::Realtime { command: RealtimeCommand::GenerateReply(_), .. })
        })
        .count();
    assert!(frame.is_some() && asked(&effects) > frame, "asked again after the frame: {effects:?}");
    assert_eq!(replies, 1, "asked once, not regenerated besides: {effects:?}");
    let handle = driver.activity.speech_scheduler.speech(speech).expect("the speech is kept");
    assert!(!handle.interrupted(), "the caller's speech is the one the new session answers");
}
