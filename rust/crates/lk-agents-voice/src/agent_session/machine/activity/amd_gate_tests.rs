use super::*;
use crate::agent_session::{
    Command, HookOutcome, Now, Sent, SpeechCall, SpeechCommand, VoiceEvent,
};
use crate::audio_recognition::TurnDetectionMode;
use crate::trace::Step;
use lk_agents_abi::voice::events::{EndOfSpeechEvent, StartOfSpeechEvent};

const RUN: AmdRunId = AmdRunId(7);

fn at(ms: u64) -> Ts {
    Ts::from_nanos(ms * 1_000_000)
}

fn now(ms: u64) -> Now {
    Now { at: at(ms), wall: std::time::UNIX_EPOCH }
}

fn driver(hosted: bool) -> VoiceDriver {
    let parts = crate::agent_session::SessionParts {
        agent: "agent".into(),
        max_unrecoverable_errors: 3,
        ..Default::default()
    };
    let bound = hosted.then_some(Duration::from_secs(5));
    let mut driver = crate::agent_session::tests::driver_with(parts, bound, now(0));
    driver.journalled = true;
    driver
}

fn host(driver: &mut VoiceDriver, ms: u64, command: Command) -> Vec<VoiceEffect> {
    let sent = Sent { origin: crate::agent_session::VoiceCommandOrigin::Host, command };
    driver.handle(Some(VoiceEvent::Host(sent)), now(ms))
}

fn answer(driver: &mut VoiceDriver, ms: u64, answer: AmdAnswer) -> Vec<VoiceEffect> {
    driver.handle(Some(VoiceEvent::Amd(answer)), now(ms))
}

/// Attach and activate the run, with no opening.
fn attached(hosted: bool) -> VoiceDriver {
    let mut driver = driver(hosted);
    host(
        &mut driver,
        0,
        Command::Amd(AmdCommand::Attach {
            run: RUN,
            dtmf_tool: super::rule_tests::dtmf_tool(),
            amd_stt: false,
            call: None,
        }),
    );
    answer(&mut driver, 0, AmdAnswer::Activated { run: RUN, opening: None });
    driver
}

fn info(turn: u64, transcript: &str) -> EndOfTurnInfo {
    EndOfTurnInfo {
        turn_id: TurnId(turn),
        owner: TurnDetectionMode::Vad,
        stt_enabled: true,
        skip_reply: false,
        new_transcript: transcript.into(),
        transcript_confidence: 0.9,
        metrics: Default::default(),
        backchannel_over_agent: false,
        late: false,
        snapshot: Default::default(),
        promoted: None,
    }
}

fn commit(driver: &mut VoiceDriver, turn: u64, transcript: &str) -> Vec<VoiceEffect> {
    driver.on_end_of_turn(driver.step.now, info(turn, transcript));
    driver.handle(None, now(driver.step.now.as_nanos() / 1_000_000))
}

fn decide(turn: u64, decision: ReplyDecision) -> AmdAnswer {
    AmdAnswer::ReplyDecided { run: RUN, turn: TurnId(turn), decision }
}

fn steps(effects: &[VoiceEffect]) -> impl Iterator<Item = &Step> {
    effects.iter().filter_map(|effect| match effect {
        VoiceEffect::Journal(step) => Some(step.as_ref()),
        _ => None,
    })
}

fn observations(effects: &[VoiceEffect]) -> Vec<&AmdObservation> {
    effects
        .iter()
        .filter_map(|effect| match effect {
            VoiceEffect::Amd { run, observation } if *run == RUN => Some(observation),
            _ => None,
        })
        .collect()
}

fn authorized(effects: &[VoiceEffect]) -> Vec<SpeechId> {
    steps(effects)
        .filter_map(|step| match step {
            Step::Speech(SpeechStep::Authorized { speech }) => Some(*speech),
            _ => None,
        })
        .collect()
}

fn replies(effects: &[VoiceEffect]) -> Vec<(TurnId, SpeechId)> {
    observations(effects)
        .into_iter()
        .filter_map(|observation| match observation {
            AmdObservation::ReplyCreated { turn, speech } => Some((*turn, *speech)),
            _ => None,
        })
        .collect()
}

fn user_items(driver: &VoiceDriver) -> Vec<String> {
    driver
        .session
        .chat_ctx
        .history()
        .filter_map(|item| item.as_message())
        .filter(|message| message.role == lk_agents_abi::Role::User)
        .filter_map(|message| message.raw_text_content())
        .collect()
}

// Catches: a gated turn replying before AMD decided it (spec §10.6: nothing but the opening
// plays before a prediction allows a reply).
#[test]
fn a_committed_turn_replies_only_after_its_decision() {
    let mut driver = attached(false);
    let effects = commit(&mut driver, 1, "hello?");
    assert!(observations(&effects).iter().any(|o| matches!(o,
        AmdObservation::TurnCommitted { turn: TurnId(1), transcript, .. } if transcript == "hello?")));
    assert!(replies(&effects).is_empty(), "no reply exists before the decision");
    assert!(steps(&effects).any(|s| *s == Step::Gate(GateStep::Parked { turn: TurnId(1) })));
    assert!(authorized(&effects).is_empty());

    let effects = answer(&mut driver, 10, decide(1, ReplyDecision::allow()));
    let created = replies(&effects);
    assert_eq!(created.len(), 1, "the allowed decision makes one reply");
    assert_eq!(authorized(&effects), [created[0].1], "the hold opens for that reply");
    assert!(observations(&effects).iter().any(|o| matches!(
        o,
        AmdObservation::TurnEnded { turn: TurnId(1), how: TurnEnd::Replied }
    )));
}

// Catches: a stale answer applied to a newer turn, and a superseded parked turn replying.
#[test]
fn a_newer_commit_denies_the_parked_turn_and_drops_its_late_decision() {
    let mut driver = attached(false);
    commit(&mut driver, 1, "first");
    let effects = commit(&mut driver, 2, "second");
    assert!(
        steps(&effects)
            .any(|s| *s == Step::Gate(GateStep::Superseded { turn: TurnId(1), speech: None }))
    );
    assert!(observations(&effects).iter().any(|o| matches!(
        o,
        AmdObservation::TurnEnded { turn: TurnId(1), how: TurnEnd::Superseded }
    )));
    assert_eq!(user_items(&driver), ["first"], "the superseded turn keeps its words, no reply");

    let effects = answer(&mut driver, 10, decide(1, ReplyDecision::allow()));
    assert!(steps(&effects).any(|s| matches!(
        s,
        Step::Gate(GateStep::AnswerDropped {
            turn: Some(TurnId(1)),
            why: AnswerDrop::NotLatest,
            ..
        })
    )));
    assert!(replies(&effects).is_empty());

    let effects = answer(&mut driver, 20, decide(2, ReplyDecision::deny()));
    assert!(replies(&effects).is_empty(), "a denied turn has no reply");
    assert_eq!(user_items(&driver), ["first", "second"]);
    assert!(steps(&effects).any(
        |s| *s == Step::Speech(SpeechStep::AuthorizationHeld(Hold::Denied { turn: TurnId(2) }))
    ));
}

// Catches: a machine-stage reply playing before `machine_silence_threshold` of silence, or a
// stale expiry opening the hold after speech restarted (spec §10.3).
#[test]
fn a_silence_hold_waits_for_silence_measured_from_the_latest_boundary() {
    let mut driver = attached(false);
    commit(&mut driver, 1, "leave a message");
    let silence =
        ReplyDecision { silence: Some(Duration::from_millis(1500)), ..ReplyDecision::allow() };
    let effects = answer(&mut driver, 100, decide(1, silence));
    let [(_, reply)] = replies(&effects)[..] else { panic!("one reply") };
    assert!(authorized(&effects).is_empty(), "held for silence");
    assert!(
        observations(&effects)
            .iter()
            .any(|o| matches!(o, AmdObservation::SilenceHold { turn: TurnId(1), held: true, .. }))
    );
    // No boundary since the commit: the hold measures from the commit, so 1.5 s.
    assert_eq!(driver.deadline(Timer::AmdSilence), Some(at(1500)));

    // The participant speaks again at 1.2 s and stops at 2 s: the deadline moves to 3.5 s.
    let start =
        StartOfSpeechEvent { at: at(1200), speech_duration: Duration::ZERO, from_vad: true };
    driver.publish_event(SessionEvent::StartOfSpeech(start));
    let effects = driver.handle(None, now(1200));
    assert!(authorized(&effects).is_empty());
    assert_eq!(driver.deadline(Timer::AmdSilence), None, "speech disarms the hold");
    let end = EndOfSpeechEvent { at: at(2000), from_vad: true, silence_duration: 0.0 };
    driver.publish_event(SessionEvent::EndOfSpeech(end));
    driver.handle(None, now(2000));
    assert_eq!(driver.deadline(Timer::AmdSilence), Some(at(3500)));

    assert!(authorized(&driver.handle(None, now(3000))).is_empty(), "the old expiry is stale");
    let effects = driver.handle(None, now(3500));
    assert_eq!(authorized(&effects), [reply]);
    assert!(
        observations(&effects)
            .iter()
            .any(|o| matches!(o, AmdObservation::SilenceHold { turn: TurnId(1), held: false, .. }))
    );
}

// Catches: AMD completion waiting on a customer hook, or the turn in its hook replying on its own
// instead of on the decision `Detach` settled.
#[test]
fn detach_settles_a_turn_still_in_its_hook() {
    let mut driver = attached(true);
    commit(&mut driver, 1, "is this a person?");
    let hook = driver.activity.pending_turn.as_ref().and_then(|p| p.hook).expect("in its hook");
    let detach =
        AmdAnswer::Detach { run: RUN, pending: ReplyDecision::deny(), cancel: DetachCancel::Keep };
    let effects = answer(&mut driver, 10, detach);
    assert!(observations(&effects).iter().any(|o| **o == AmdObservation::Detached { run: RUN }));
    assert!(
        steps(&effects)
            .any(|s| *s == Step::Gate(GateStep::Settled { turn: TurnId(1), allow: false }))
    );
    assert!(driver.activity.gate.is_none(), "the gate is removed without waiting for the hook");

    driver.on_hook_done(at(20), hook, HookOutcome::Returned);
    let effects = driver.handle(None, now(20));
    assert!(authorized(&effects).is_empty(), "the settled decision denies the reply");
    assert_eq!(user_items(&driver), ["is this a person?"]);

    // Later turns are not gated.
    let effects = commit(&mut driver, 2, "hello");
    assert!(observations(&effects).is_empty());
}

// Catches: a host `say()` slipping past the hold, and the opening being held like any reply.
#[test]
fn only_the_opening_bypasses_the_hold() {
    let mut driver = driver(false);
    driver.session.outputs.audio = None;
    host(
        &mut driver,
        0,
        Command::Amd(AmdCommand::Attach {
            run: RUN,
            dtmf_tool: super::rule_tests::dtmf_tool(),
            amd_stt: false,
            call: None,
        }),
    );
    let say = SpeechCommand::Say {
        text: crate::agent_session::SayText::Text("hi".into()),
        allow_interruptions: None,
        add_to_chat_ctx: true,
        call: None::<SpeechCall>,
    };
    let effects = host(&mut driver, 0, Command::Speech(say));
    let created = observations(&effects).into_iter().any(|o| {
        matches!(o, AmdObservation::Speech { event: SpeechObservation::Created { .. }, .. })
    });
    assert!(!created, "say() is refused while AMD is attached: no speech exists");

    let opening = Some(Opening::Text("Hello, this is Ada calling.".into()));
    let effects = answer(&mut driver, 5, AmdAnswer::Activated { run: RUN, opening });
    let bypass: Vec<_> = steps(&effects)
        .filter_map(|s| match s {
            Step::Gate(GateStep::OpeningBypass { speech }) => Some(*speech),
            _ => None,
        })
        .collect();
    assert_eq!(bypass.len(), 1);
    assert_eq!(authorized(&effects), bypass, "the opening plays under the entry hold");
}

// Catches: a stage the application handles still speaking a reply that has not played yet
// (spec §12.3).
#[test]
fn detach_that_forbids_replies_cancels_the_unplayed_reply() {
    let mut driver = attached(false);
    commit(&mut driver, 1, "please hold");
    let silence = ReplyDecision { silence: Some(Duration::from_secs(5)), ..ReplyDecision::allow() };
    let effects = answer(&mut driver, 10, decide(1, silence));
    let [(_, reply)] = replies(&effects)[..] else { panic!("one reply") };
    assert!(authorized(&effects).is_empty(), "held for silence");
    let detach = AmdAnswer::Detach {
        run: RUN,
        pending: ReplyDecision::deny(),
        cancel: DetachCancel::ForbidReplies,
    };
    let effects = answer(&mut driver, 20, detach);
    assert!(authorized(&effects).is_empty(), "the held reply never plays");
    assert!(driver.activity.speech_scheduler.speech(reply).is_none_or(|h| h.interrupted()));
    assert!(
        observations(&effects)
            .iter()
            .any(|o| matches!(o, AmdObservation::SilenceHold { turn: TurnId(1), held: false, .. }))
    );
}

// Catches: a decision that arrives while the customer hook runs being lost, or applied before the
// hook's edits.
#[test]
fn a_decision_before_the_hook_returns_is_applied_when_it_returns() {
    let mut driver = attached(true);
    commit(&mut driver, 1, "who is this");
    let hook = driver.activity.pending_turn.as_ref().and_then(|p| p.hook).expect("in its hook");
    let effects = answer(&mut driver, 10, decide(1, ReplyDecision::allow()));
    assert!(steps(&effects).any(|s| *s
        == Step::Gate(GateStep::Decided { run: RUN, turn: TurnId(1), allow: true })));
    assert!(replies(&effects).is_empty(), "the hook still runs");

    let edit = crate::agent::TurnEdit::UpdateUserMessage("who is this?".into());
    driver.on_hook_done(at(20), hook, HookOutcome::Turn(vec![edit]));
    let effects = driver.handle(None, now(20));
    let [(TurnId(1), reply)] = replies(&effects)[..] else { panic!("one reply for turn 1") };
    assert_eq!(authorized(&effects), [reply]);
    assert_eq!(user_items(&driver), ["who is this?"], "the hook's edit reaches the record");
}
