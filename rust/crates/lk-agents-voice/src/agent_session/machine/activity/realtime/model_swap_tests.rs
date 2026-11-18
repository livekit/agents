use super::*;
use crate::agent_session::effects::VoiceEffect;
use crate::agent_session::machine::activity::handoff::HandoffEvent;
use crate::agent_session::machine::activity::realtime::{REQUEST_TIMEOUT, RealtimeWake, Request};
use crate::agent_session::machine::activity::scheduling::VoiceRunParticipation;
use crate::agent_session::tests::{agent_update, realtime_driver};
use crate::agent_session::{Answer, Now, RequestId, VoiceEvent};
use lk_agents_abi::llm::realtime as rt;
use lk_agents_abi::{ChatMessage, RealtimeCommand, Role};
use std::time::{Duration, UNIX_EPOCH};

const OLD: ResourceId = ResourceId(1);
const NEW: ResourceId = ResourceId(40);
const NEW_SINK: ResourceId = ResourceId(41);

fn model(id: u64, instance: usize) -> RealtimeModelRef {
    RealtimeModelRef {
        id: ResourceId(id),
        provider: lk_agents_abi::ProviderId::of_label("realtime"),
        capabilities: Default::default(),
        instance,
    }
}

/// A session on model `1`, which said `Hello.` to the person.
fn on_model_one() -> VoiceDriver {
    let mut driver = realtime_driver();
    driver.activity.realtime_activity_mut().expect("realtime").model = Some(model(10, 1));
    let hello = ChatMessage::text("hello", Role::Assistant, "Hello.", UNIX_EPOCH);
    driver.commit_item(hello.into(), crate::agent::Source::HostCtx, crate::agent::Visibility::Both);
    driver.settle_step();
    driver
}

fn switch_to(driver: &mut VoiceDriver, agent: &str, model: RealtimeModelRef) -> Vec<VoiceEffect> {
    let mut update = agent_update(agent, Some(RequestId(7)));
    update.realtime = Some(model);
    driver.handoff_step(HandoffEvent::Asked { update: Box::new(update), run: None });
    driver.settle_step()
}

fn at(seconds: u64) -> Now {
    Now { at: Ts::ZERO.saturating_add(Duration::from_secs(seconds)), wall: UNIX_EPOCH }
}

fn opened(open: u64) -> VoiceEvent {
    let session = RealtimeRef {
        id: NEW,
        provider: lk_agents_abi::ProviderId::of_label("realtime"),
        capabilities: Default::default(),
    };
    VoiceEvent::RealtimeOpened { open, outcome: Ok((session, NEW_SINK)) }
}

fn handoff_answer(effects: &[VoiceEffect]) -> Option<Result<(), HandoffRefused>> {
    effects.iter().find_map(|effect| match effect {
        VoiceEffect::Answer(RequestId(7), Answer::Handoff(result)) => Some(*result),
        _ => None,
    })
}

fn sent_to(effects: &[VoiceEffect], to: ResourceId) -> Vec<&RealtimeCommand> {
    effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Realtime { session, command } if *session == to => Some(command),
            _ => None,
        })
        .collect()
}

// Catches: an agent on another realtime model landing on the old connection, or the new session
// starting without the conversation, so the new model does not continue it.
#[test]
fn a_switch_to_another_realtime_model_lands_on_its_own_session_with_the_conversation() {
    let mut driver = on_model_one();
    let asked = switch_to(&mut driver, "billing", model(30, 2));
    assert!(
        asked.iter().any(|e| matches!(e, VoiceEffect::OpenRealtime { open: 1, model, .. }
            if *model == ResourceId(30))),
        "the arriving model opens its session: {asked:?}"
    );
    assert_eq!(handoff_answer(&asked), None, "the switch waits for the new session");
    assert_eq!(driver.activity.agent_name(), "agent", "the leaving agent stays connected");

    let landed = driver.handle(Some(opened(1)), at(0));
    assert_eq!(handoff_answer(&landed), Some(Ok(())));
    assert_eq!(driver.activity.agent_name(), "billing");
    assert_eq!(driver.activity.realtime().map(|s| s.id), Some(NEW));
    assert!(
        landed.iter().any(|e| matches!(e, VoiceEffect::CloseRealtimeSession(id) if *id == OLD)),
        "the old connection closes: {landed:?}"
    );
    assert!(
        landed.iter().any(|e| matches!(e, VoiceEffect::UpdateAudioInputRouting(
            crate::data_plane::AudioInputRouterCommand::Install { channel: AudioChannel::Realtime, sink }
        ) if *sink == NEW_SINK)),
        "the caller's audio goes to the new session"
    );
    let frame = sent_to(&landed, NEW);
    let history = frame.iter().find_map(|command| match command {
        RealtimeCommand::UpdateChatCtx(update) => Some(&update.chat_ctx),
        _ => None,
    });
    let history = history.expect("the new session is given the conversation");
    assert!(
        history.items.iter().any(|item| item.id() == "hello"),
        "what the old model said is carried: {history:?}"
    );
    assert!(sent_to(&landed, OLD).iter().all(|c| !matches!(c, RealtimeCommand::UpdateChatCtx(_))));
}

// Catches: a switch back to the model the session runs closing and reopening its connection.
#[test]
fn a_switch_to_the_connected_model_keeps_the_connection() {
    let mut driver = on_model_one();
    let effects = switch_to(&mut driver, "billing", model(30, 1));
    assert_eq!(handoff_answer(&effects), Some(Ok(())));
    assert_eq!(driver.activity.realtime().map(|s| s.id), Some(OLD));
    assert!(!effects.iter().any(|e| matches!(
        e,
        VoiceEffect::OpenRealtime { .. } | VoiceEffect::CloseRealtimeSession(_)
    )));
}

// Catches: a late event of the replaced socket (its end as it closes) ending the new session.
#[test]
fn the_replaced_sessions_late_events_never_reach_the_new_one() {
    let mut driver = on_model_one();
    switch_to(&mut driver, "billing", model(30, 2));
    driver.handle(Some(opened(1)), at(0));
    driver.handle(
        Some(VoiceEvent::RealtimeSession { session: OLD, wake: RealtimeWake::Ended }),
        at(0),
    );
    let state = &driver.activity.realtime_activity().expect("realtime").state;
    assert!(!state.ended(), "the old socket's end ended its successor");
}

// Catches: a request the old session owed never settling, or its timer, still armed, failing a
// request of the new session that reused its key.
#[test]
fn the_old_sessions_requests_settle_and_its_timers_never_fire_on_the_new_one() {
    let mut driver = on_model_one();
    let ask = Request::GenerateReply { instructions: None, tool_choice: None, read_back: false };
    let old_speech = driver
        .realtime_request(Ts::ZERO, ask.clone(), None, None, VoiceRunParticipation::Outside)
        .expect("a speech for the reply");
    driver.settle_step();
    let session = RealtimeRef {
        id: NEW,
        provider: lk_agents_abi::ProviderId::of_label("realtime"),
        capabilities: Default::default(),
    };
    driver.swap_realtime_session(OpenedSession { session, sink: NEW_SINK, model: model(30, 2) });
    let swapped = driver.handle(None, at(5));
    assert!(
        swapped.iter().any(|e| matches!(e, VoiceEffect::CloseRealtimePlayers)),
        "the old reply's player ends with its session"
    );
    assert!(
        driver.activity.realtime_activity().is_some_and(|rt| !rt.players.contains(&old_speech))
    );

    let speech = driver
        .realtime_request(at(5).at, ask, None, None, VoiceRunParticipation::Outside)
        .expect("a speech for the new session's reply");
    driver.settle_step();
    let late =
        driver.handle(None, Now { at: Ts::ZERO.saturating_add(REQUEST_TIMEOUT), wall: UNIX_EPOCH });
    assert!(
        !late.iter().any(|e| matches!(e, VoiceEffect::RealtimePlayer { speech: s, event }
            if *s == speech && matches!(**event, rt::Event::GenerateReplyFailed(_)))),
        "the old request's timer failed the new session's request: {late:?}"
    );
    assert!(driver.activity.realtime_activity().is_some_and(|rt| rt.players.contains(&speech)));
}

// Catches: a model that could not open leaving the switch hanging, or the session on no model.
#[test]
fn a_model_that_fails_to_open_refuses_the_switch_and_keeps_the_connection() {
    let mut driver = on_model_one();
    switch_to(&mut driver, "billing", model(30, 2));
    let failed = lk_agents_abi::errors::ProviderError::Local(lk_agents_abi::errors::ApiError::new(
        lk_agents_abi::errors::ErrorKind::Server,
        "no socket",
    ));
    let effects =
        driver.handle(Some(VoiceEvent::RealtimeOpened { open: 1, outcome: Err(failed) }), at(0));
    assert_eq!(handoff_answer(&effects), Some(Err(HandoffRefused::SetupFailed)));
    assert_eq!(driver.activity.agent_name(), "agent");
    assert_eq!(driver.activity.realtime().map(|s| s.id), Some(OLD));
    assert!(!driver.activity.in_window(), "no switch is left holding new turns");
    assert!(!driver.activity.speech_scheduler.is_paused(), "the agent that stays speaks again");
}

// Catches: a session a closed call's switch opened late left running.
#[test]
fn a_session_opened_after_the_switch_was_dropped_is_closed_unused() {
    let mut driver = on_model_one();
    switch_to(&mut driver, "billing", model(30, 2));
    driver.handoff_step(HandoffEvent::Closing);
    driver.settle_step();
    let effects = driver.handle(Some(opened(1)), at(0));
    assert!(
        effects.iter().any(|e| matches!(e, VoiceEffect::CloseRealtimeSession(id) if *id == NEW))
    );
    assert_eq!(driver.activity.realtime().map(|s| s.id), Some(OLD));
}

// Catches: an agent switch that changes the kind of model, or the capabilities that shaped the
// session's audio pipeline, accepted and leaving the pipeline wrong for the new model.
#[test]
fn a_switch_keeps_the_kind_of_model_and_its_pipeline_shape() {
    let caps =
        RealtimeCapabilities { audio_output: true, turn_detection: true, ..Default::default() };
    assert_eq!(model_swap_allowed(None, None), Ok(()));
    assert_eq!(model_swap_allowed(Some(&caps), Some(&caps)), Ok(()));
    let richer = RealtimeCapabilities { supports_say: true, ..caps };
    assert_eq!(model_swap_allowed(Some(&caps), Some(&richer)), Ok(()));
    assert_eq!(model_swap_allowed(Some(&caps), None), Err(ModelSwapRefused::KindChanged));
    assert_eq!(model_swap_allowed(None, Some(&caps)), Err(ModelSwapRefused::KindChanged));
    for other in [
        RealtimeCapabilities { audio_output: false, ..caps },
        RealtimeCapabilities { turn_detection: false, ..caps },
        RealtimeCapabilities { can_disable_turn_detection: true, ..caps },
    ] {
        assert_eq!(
            model_swap_allowed(Some(&caps), Some(&other)),
            Err(ModelSwapRefused::PipelineChanged)
        );
    }
}
