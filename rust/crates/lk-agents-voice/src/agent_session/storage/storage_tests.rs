use super::*;
use crate::agent_session::machine::activity::scheduling::{SpeechKind, VoiceRunParticipation};
use crate::agent_session::tests::driver;
use crate::events::SessionEvent;
use crate::speech_handle::SpeechState;
use crate::trace::{SpeechStep, Step};
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_core::agent_run::RequestId;

#[test]
fn reloading_history_does_not_publish_a_private_realtime_placeholder() {
    use crate::agent::{Source, Visibility};

    let mut driver = driver();
    driver.commit_item(
        ChatMessage::text("placeholder", Role::User, "", std::time::UNIX_EPOCH).into(),
        Source::Remote { item_id: "placeholder".into() },
        Visibility::Model,
    );
    let foreign: ChatItem =
        ChatMessage::text("text", Role::Assistant, "Sent by text.", std::time::UNIX_EPOCH).into();
    driver.settle_step();
    driver.take_outbox();
    driver.storage_reloaded(
        Vec::new(),
        VoiceHistory { items: vec![foreign.clone()], stored: vec![foreign], revision: (1, 1) },
    );
    let effects = driver.settle_step();
    let history = effects
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::Emit(SessionEvent::HistoryReloaded { items }) => Some(items),
            _ => None,
        })
        .expect("the foreign text reloads history");
    assert!(history.iter().any(|item| item.id() == "text"));
    assert!(!history.iter().any(|item| item.id() == "placeholder"));
    assert!(
        driver.session.chat_ctx.items().any(|item| item.id() == "placeholder"),
        "the realtime model keeps its private placeholder until its final transcript"
    );
}

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

#[test]
fn acquiring_storage_uses_the_conversation_record_instead_of_a_private_prompt() {
    use crate::agent::{Source, Visibility};
    use crate::agent_session::effects::SpeechHistory;
    use crate::agent_session::machine::activity::speech::SpeechSpec;
    use crate::generation::ReplySource;
    use lk_agents_abi::ChatContext;

    let mut driver = driver();
    driver.session.storage.enabled = true;
    let agent = driver.activity.agent_name().to_owned();
    let message = |id, text| ChatMessage::text(id, Role::Assistant, text, std::time::UNIX_EPOCH);
    let _ = driver.session.chat_ctx.commit(
        message("delivered", "Hello.").into(),
        Source::HostCtx,
        Visibility::Both,
        &agent,
    );
    let expected = driver.checkpoint_history().items;
    let mut spec = SpeechSpec::new(SpeechKind::GenerateReply, ReplySource::Llm, true);
    spec.chat_ctx = Some(ChatContext::new(vec![message("private", "Model reminder.").into()]));
    driver.start_speech(Ts::ZERO, spec);
    let effects = driver.take_outbox();
    let (context, history) = effects
        .iter()
        .find_map(|effect| match effect {
            VoiceEffect::StartSpeech { context, history, .. } => Some((context, history)),
            _ => None,
        })
        .expect("the reply starts");
    let SpeechHistory::VoicePartition { history } = history else {
        panic!("the bound reply acquires its partition")
    };
    assert_eq!(history.items, expected);
    assert!(context.request.chat_ctx.items.iter().any(|item| item.id() == "private"));
    assert!(context.replan.is_some(), "storage reload precedes the model's request");
}

#[test]
fn an_older_receipt_cannot_replace_the_conversation_or_its_read_baseline() {
    use crate::agent::{Source, Visibility};

    let mut driver = driver();
    let agent = driver.activity.agent_name().to_owned();
    let message = |text| {
        ChatItem::Message(ChatMessage::text("text", Role::Assistant, text, std::time::UNIX_EPOCH))
    };
    let original = message("Original.");
    let current = message("Edited by text.");
    let _ =
        driver.session.chat_ctx.commit(original.clone(), Source::HostCtx, Visibility::Both, &agent);
    driver.storage_reloaded(
        vec![original.clone()],
        VoiceHistory {
            items: vec![current.clone()],
            stored: vec![current.clone()],
            revision: (1, 2),
        },
    );
    driver.take_outbox();
    let expected = driver.checkpoint_history().items;
    assert!(expected.contains(&current), "the newer receipt was accepted");
    driver.storage_reloaded(
        vec![current.clone()],
        VoiceHistory { items: vec![original.clone()], stored: vec![original], revision: (1, 1) },
    );
    assert_eq!(driver.checkpoint_history().items, expected);
    assert_eq!(driver.checkpoint_history().stored, [current]);
    assert_eq!(driver.checkpoint_history().revision, (1, 2));
    assert!(driver.take_outbox().is_empty(), "an old receipt publishes no replacement history");
}
