use super::*;
use crate::agent_session::machine::activity::scheduling::{SpeechKind, VoiceRunParticipation};
use crate::agent_session::tests::driver;
use crate::speech_handle::SpeechState;
use crate::trace::{SpeechStep, Step};
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_core::agent_run::RequestId;

/// A bound driver with one scheduled speech holding the partition, journalled.
fn speaking(speech: SpeechId) -> VoiceDriver {
    let mut driver = driver();
    driver.journalled = true;
    driver.session.storage.enabled = true;
    assert!(driver.schedule_speech(
        Ts::ZERO,
        speech,
        SpeechState::new(true),
        SpeechKind::GenerateReply,
        true,
        true,
        VoiceRunParticipation::Joins,
    ));
    driver.storage_started(speech);
    driver.storage_answered(Ts::ZERO, speech, StorageRequest::Acquire, Ok(()));
    driver.take_outbox();
    driver
}

fn outcome() -> TurnOutcome {
    TurnOutcome { forwarded_text: "heard".into(), assistant_message: None, cut_at: None }
}

fn checkpoint_of(effects: &[VoiceEffect]) -> u64 {
    effects
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::StorageCheckpoint { checkpoint, .. } => Some(*checkpoint),
            _ => None,
        })
        .expect("a checkpoint was asked")
}

fn failed(effects: &[VoiceEffect], speech: SpeechId) -> usize {
    effects
        .iter()
        .filter(|effect| {
            matches!(effect, VoiceEffect::Journal(step)
                if matches!(step.as_ref(), Step::Speech(SpeechStep::ReplyFailed { speech: s, .. })
                    if *s == speech))
        })
        .count()
}

fn saves(effects: &[VoiceEffect]) -> usize {
    effects
        .iter()
        .filter(|effect| {
            matches!(effect, VoiceEffect::StorageCommit { .. } | VoiceEffect::StorageAbort(_))
        })
        .count()
}

fn unavailable() -> StorageError {
    StorageError::Unavailable("database is unreachable".into())
}

// Catches (seeds 8, 150, 375): a request a speech's turn waited on failed (its commit, the
// checkpoint its reply's next step waited on, or the checkpoint its reported commit waited on).
// The call closed with the storage error, but the speech, waiting on that answer alone, was ended
// only by the 5 s interruption watchdog. Every failed answer ends the speech at once, exactly
// once, whatever the turn was waiting for, and a later report of the turn saves nothing.
#[test]
fn a_failed_storage_answer_ends_the_speech_at_once() {
    let speech = SpeechId(12);
    type Ask = fn(&mut VoiceDriver, SpeechId) -> StorageRequest;
    let cases: [(&str, Ask); 3] = [
        ("its commit", |driver, speech| {
            driver.storage_settle(Ts::ZERO, speech, outcome(), false);
            assert_eq!(saves(&driver.take_outbox()), 1, "the reported turn commits");
            StorageRequest::Commit
        }),
        ("the checkpoint its next step waits on", |driver, speech| {
            let next_step = AgentRunEffect::PlanRequest {
                request: RequestId(1),
                message_id: "reply".into(),
                pending: Vec::new(),
                tools: Vec::new(),
                tools_forbidden: false,
            };
            driver.checkpoint_reply_effect(speech, next_step);
            StorageRequest::Checkpoint(checkpoint_of(&driver.take_outbox()))
        }),
        ("the checkpoint its commit waits on", |driver, speech| {
            let heard = ChatMessage::text("reply", Role::Assistant, "heard", std::time::UNIX_EPOCH);
            driver.storage_progress(speech, heard);
            let checkpoint = checkpoint_of(&driver.take_outbox());
            driver.storage_settle(Ts::ZERO, speech, outcome(), false);
            assert_eq!(saves(&driver.take_outbox()), 0, "the commit waits on the checkpoint");
            StorageRequest::Checkpoint(checkpoint)
        }),
    ];
    for (waits_on, ask) in cases {
        let mut driver = speaking(speech);
        let request = ask(&mut driver, speech);
        driver.storage_answered(Ts::ZERO, speech, request, Err(unavailable()));
        let effects = driver.take_outbox();
        assert_eq!(failed(&effects, speech), 1, "{waits_on}: the speech ends, failed, at once");
        assert!(driver.session.closing.is_some(), "{waits_on}: the bound call closes");
        driver.storage_settle(Ts::ZERO, speech, outcome(), true);
        let effects = driver.take_outbox();
        assert_eq!(saves(&effects), 0, "{waits_on}: a turn ends once");
        assert_eq!(failed(&effects, speech), 0, "{waits_on}: a speech ends once");
    }
}

// Catches (seed 470): a realtime reply was heard while its partition was still being taken; the
// words' checkpoint went to a store no turn owned yet, which refused it, and the speech failed.
// What a turn would save waits until the partition is taken.
#[test]
fn a_turn_saves_nothing_before_its_partition_is_taken() {
    let speech = SpeechId(15);
    let mut driver = speaking(speech);
    let other = SpeechId(16);
    driver.storage_started(other);
    let heard = ChatMessage::text("heard", Role::Assistant, "heard", std::time::UNIX_EPOCH);
    driver.storage_progress(other, heard);
    assert!(
        !driver.take_outbox().iter().any(|e| matches!(e, VoiceEffect::StorageCheckpoint { .. })),
        "nothing is checkpointed before the partition is taken"
    );
    driver.storage_answered(Ts::ZERO, other, StorageRequest::Acquire, Ok(()));
    checkpoint_of(&driver.take_outbox());
}
