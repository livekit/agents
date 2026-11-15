use lk_agents_abi::{ChatItem, Role, SpeechEvent, SpeechEventType};
use lk_agents_voice::agent_session::machine::activity::scheduling::{SpeechId, SpeechKind};
use lk_agents_voice::events::{SessionEvent, SpeechOutcome, UserTranscriptEvent};
use lk_agents_voice::trace::{
    GenerationStep, InputStep, InterruptCause, InterruptStep, SpeechStep, Step, TurnStep,
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{Input, Ref, UserText},
    record::{Kind, Observed, Recording, StorageOperation, StorageStep},
};

pub struct CorrectionReplacesTheWords;

impl Rule for CorrectionReplacesTheWords {
    const CONTRACT: Contract = Contract {
        id: "correction_replaces_the_words",
        guarantee: "The final of an utterance whose interim a committed turn took replaces those words in that turn's user message, as no turn of its own, and an empty final changes nothing; a reply none of which was heard is made again from the corrected record, and one partly heard plays on.",
        examples: &[corrected_before_the_reply_played, corrected_while_the_reply_plays],
        counterexamples: &[
            correction_dropped,
            empty_final_corrected,
            restarted_after_playback,
            correction_not_saved,
        ],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // A realtime model hears the audio itself: the recogniser's words are not its record.
        if r.opened().iter().any(|o| o.kind == Kind::Realtime) {
            return;
        }
        let messages = user_messages(r);
        for (at, step) in r.journal() {
            let Step::Emitted(SessionEvent::UserInputCorrected { item_id, text, .. }) = step else {
                continue;
            };
            check.expect(Subject::Session, !text.is_empty(), "an empty final corrected words");
            let Some(held) = item_id else { continue };
            corrected(r, check, &messages, held, *at, text);
        }
        every_late_final_corrects(r, check);
    }
}

/// A correction of `held`: its last version reads `text`, the thread saves that, no user message
/// of its own appears, and its reply is made again only while none of it was heard.
fn corrected(
    r: &Recording,
    check: &mut Check,
    messages: &[(Observed, String, String)],
    held: &str,
    at: Observed,
    text: &str,
) {
    let last = messages.iter().rev().find(|(_, id, _)| id == held);
    check.expect(
        Subject::Session,
        last.is_some_and(|(_, _, said)| said.contains(text)),
        format!("the correction {text:?} never reached {held}"),
    );
    // A turn saved after the correction saves the whole conversation: its thread holds the same
    // words, unless a fault cost the store its writes.
    if saved_after(r, at)
        && let Some(saved) = saved_text(r, held)
    {
        check.expect(
            Subject::Session,
            saved.contains(text),
            format!("the thread saved {held} as {saved:?}, not with {text:?}"),
        );
    }
    let within = |o: &Observed| o.seq >= at.seq && o.at_ms == at.at_ms;
    let first_seen = |id: &str| messages.iter().find(|(_, m, _)| m == id).map(|(o, ..)| *o);
    check.expect(
        Subject::Session,
        !messages.iter().any(|(o, id, _)| within(o) && first_seen(id) == Some(*o)),
        format!("the correction {text:?} added a user message of its own"),
    );
    reply_after(r, check, messages, held, at, text);
}

/// Each non-empty final of an utterance whose interim a turn already committed with yields a
/// correction with its words, unless the record moved on first: another utterance's words, a
/// newer turn, a swapped recogniser, a handoff or the close.
fn every_late_final_corrects(r: &Recording, check: &mut Check) {
    let journal = r.journal();
    let applied: Vec<_> = r.inputs_view().accepted().collect();
    for (i, late) in applied.iter().enumerate() {
        let Input::SttFinal { utterance, text: UserText(text), .. } = &late.input else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let since = applied[..i].iter().rev().find_map(|a| match &a.input {
            Input::SttFinal { utterance: u, .. } if u == utterance => Some(a.at.seq),
            _ => None,
        });
        // The session made the utterance's interim final before the provider did.
        let Some(promoted) = journal.iter().find_map(|(o, s)| match s {
            Step::Emitted(SessionEvent::UserInputTranscribed(t))
                if t.is_final
                    && t.ev.utterance == Some(u64::from(utterance.nth))
                    && since.is_none_or(|since| o.seq > since)
                    && o.seq < late.at.seq =>
            {
                Some(*o)
            }
            _ => None,
        }) else {
            continue;
        };
        let between = |o: &Observed| o.seq > promoted.seq && o.seq < late.at.seq;
        let other_words = applied.iter().any(|a| {
            between(&a.at)
                && matches!(&a.input,
                    Input::SttInterim { utterance: u, .. }
                    | Input::SttPreflight { utterance: u, .. }
                    | Input::SttFinal { utterance: u, .. } if u != utterance)
        });
        let turns = journal
            .iter()
            .filter(|(o, s)| {
                o.seq > promoted.seq && matches!(s, Step::Turn(TurnStep::EndOfTurn { .. }))
            })
            .filter(|(o, _)| o.seq < late.at.seq)
            .count();
        let moved_on = journal.iter().any(|(o, s)| {
            o.seq > promoted.seq
                && matches!(
                    s,
                    Step::Input(InputStep::LegItemDiscarded { .. })
                        | Step::Handoff(_)
                        | Step::Emitted(SessionEvent::Close { .. })
                )
        });
        if other_words || turns > 1 || moved_on {
            continue;
        }
        let corrects = journal.iter().any(|(o, s)| {
            o.seq > late.at.seq
                && matches!(s, Step::Emitted(SessionEvent::UserInputCorrected { text: t, .. })
                    if t == text.trim())
        });
        check.expect(
            Subject::Session,
            corrects,
            format!("the final {text:?} of words a turn took never corrected them"),
        );
    }
}

/// Whether a turn's save was asked after `at` and answered.
fn saved_after(r: &Recording, at: Observed) -> bool {
    let storage = r.host_view().storage();
    storage.iter().any(|(o, step)| {
        o.seq > at.seq
            && matches!(
                step,
                StorageStep::Answered { operation: StorageOperation::Settle, result: Ok(()), .. }
            )
    }) && storage.iter().any(|(o, step)| {
        o.seq > at.seq
            && matches!(step, StorageStep::Requested { operation: StorageOperation::Settle, .. })
    })
}

/// The text the thread saved `id` with, from a store that met no fault.
fn saved_text(r: &Recording, id: &str) -> Option<String> {
    let storage = r.host_view().storage();
    let faulted = storage.iter().any(|(_, step)| {
        matches!(step, StorageStep::PartitionSettled { faulted: true, .. })
            || matches!(step, StorageStep::Fault { .. })
    });
    if faulted {
        return None;
    }
    storage.iter().rev().find_map(|(_, step)| match step {
        StorageStep::Saved { items } => items.iter().find_map(|item| match item {
            ChatItem::Message(m) if m.id == id && m.role == Role::User => m.raw_text_content(),
            _ => None,
        }),
        _ => None,
    })
}

/// Every version of every user message the host was shown: when, its id and its text.
fn user_messages(r: &Recording) -> Vec<(Observed, String, String)> {
    r.journal()
        .iter()
        .filter_map(|(o, step)| match step {
            Step::Emitted(SessionEvent::ConversationItemAdded { item, .. }) => match &**item {
                ChatItem::Message(m) if m.role == Role::User => {
                    Some((*o, m.id.clone(), m.raw_text_content().unwrap_or_default()))
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// The reply made with the corrected turn: made again when none of it was heard, left playing
/// when some was.
fn reply_after(
    r: &Recording,
    check: &mut Check,
    messages: &[(Observed, String, String)],
    held: &str,
    at: Observed,
    text: &str,
) {
    let within = |o: &Observed| o.seq >= at.seq && o.at_ms == at.at_ms;
    let Some((committed, ..)) = messages.iter().find(|(_, id, _)| id == held) else { return };
    // A newer turn since: the reply being made is that one's.
    if messages.iter().any(|(o, id, _)| o.seq > committed.seq && o.seq < at.seq && id != held) {
        return;
    }
    let journal = r.journal();
    // Made in the instant the message was committed, just before it.
    let made_with = |o: &Observed| o.at_ms == committed.at_ms && o.seq < committed.seq;
    let reply = journal.iter().rev().find_map(|(o, step)| match step {
        Step::Emitted(SessionEvent::SpeechCreated {
            speech,
            kind: SpeechKind::TurnReply | SpeechKind::Preemptive,
            ..
        }) if made_with(o) => Some(*speech),
        _ => None,
    });
    let Some(reply) = reply else { return };
    let before = |o: &Observed| o.seq < at.seq;
    let done = journal.iter().any(|(o, s)| {
        before(o) && matches!(s, Step::Emitted(SessionEvent::SpeechDone { speech, .. }) if *speech == reply)
    });
    let interrupted_before = journal.iter().any(|(o, s)| {
        before(o)
            && matches!(s, Step::Interrupt(InterruptStep::Interrupted { speech, .. }) if *speech == reply)
    });
    if done || interrupted_before {
        return;
    }
    let played = journal.iter().any(|(o, s)| {
        before(o)
            && matches!(s, Step::Speech(SpeechStep::AgentStartedSpeaking { speech }) if *speech == reply)
    });
    let cut_now = journal.iter().any(|(o, s)| {
        within(o)
            && matches!(s, Step::Interrupt(InterruptStep::Interrupted { speech, cause: InterruptCause::Superseded, .. }) if *speech == reply)
    });
    if played {
        check.expect(
            Subject::Speech(reply.0),
            !cut_now,
            "a reply partly heard was cut for a correction",
        );
        return;
    }
    check.expect(
        Subject::Speech(reply.0),
        cut_now,
        format!("a reply none of which was heard was not made again for {text:?}"),
    );
    let again = journal.iter().find_map(|(o, s)| match s {
        Step::Emitted(SessionEvent::SpeechCreated {
            speech, kind: SpeechKind::TurnReply, ..
        }) if within(o) => Some(*speech),
        _ => None,
    });
    let Some(again) = again else {
        check.expect(Subject::Speech(reply.0), false, "no reply was made again");
        return;
    };
    // Its model request reads the corrected message.
    if let Some((asked, ids)) = journal.iter().find_map(|(o, s)| match s {
        Step::Generation(GenerationStep::ChatRequested { speech, ctx_ids, .. })
            if *speech == again =>
        {
            Some((*o, ctx_ids))
        }
        _ => None,
    }) {
        let read = messages.iter().rev().find(|(o, id, _)| id == held && o.seq <= asked.seq);
        check.expect(
            Subject::Speech(again.0),
            ids.iter().any(|id| id == held) && read.is_some_and(|(_, _, said)| said.contains(text)),
            format!("the reply made again did not read {text:?}"),
        );
    }
}

fn user(id: &str, text: &str) -> Step {
    Step::Emitted(SessionEvent::ConversationItemAdded {
        item: Box::new(ChatItem::Message(lk_agents_abi::ChatMessage::text(
            id,
            Role::User,
            text,
            std::time::UNIX_EPOCH,
        ))),
        agent: "agent".into(),
        source: lk_agents_voice::agent::Source::UserTurn {
            turn: lk_agents_voice::audio_recognition::TurnId(1),
        },
        visibility: lk_agents_voice::agent::Visibility::Both,
    })
}

/// The session made the interim "hello" of utterance 0 final as its turn committed.
fn promoted() -> Step {
    let mut ev = SpeechEvent::new(SpeechEventType::FinalTranscript);
    ev.alternatives = vec![lk_agents_abi::SpeechData::new("en", "hello")];
    ev.utterance = Some(0);
    Step::Emitted(SessionEvent::UserInputTranscribed(Box::new(UserTranscriptEvent {
        ev,
        is_final: true,
        speaking: None,
        item_id: None,
        segment: "SG_1".into(),
    })))
}

fn corrected_event(text: &str) -> Step {
    Step::Emitted(SessionEvent::UserInputCorrected {
        segment: "SG_1".into(),
        item_id: Some("item_1".into()),
        text: text.into(),
    })
}

fn created(speech: u64) -> Step {
    Step::Emitted(SessionEvent::SpeechCreated {
        speech: SpeechId(speech),
        kind: SpeechKind::TurnReply,
        user_initiated: false,
    })
}

fn interrupted(speech: u64) -> Step {
    Step::Interrupt(InterruptStep::Interrupted {
        speech: SpeechId(speech),
        cause: InterruptCause::Superseded,
        force: true,
    })
}

fn requested(speech: u64, ids: &[&str]) -> Step {
    Step::Generation(GenerationStep::ChatRequested {
        speech: SpeechId(speech),
        step: 0,
        ctx_hash: 0,
        tools: Vec::new(),
        instructions_hash: 0,
        ctx_ids: ids.iter().map(|id| (*id).to_owned()).collect(),
        tool_choice: None,
    })
}

fn said(text: &str, final_: bool) -> Input {
    let (utterance, text) = (Ref::new(0), UserText(text.into()));
    if final_ {
        Input::SttFinal { utterance, text, speaker: None }
    } else {
        Input::SttInterim { utterance, text, speaker: None }
    }
}

/// The turn committed with the interim "hello", its reply not yet heard.
fn committed() -> Authored {
    Authored::new()
        .sent(said("hello", false))
        .journal(promoted())
        .journal_at_once(created(1))
        .journal_at_once(user("item_1", "hello"))
}

/// The late final rewrites the message and the reply is made again from it.
fn corrected_before_the_reply_played() -> Recording {
    committed()
        .sent(said("hello there", true))
        .journal(corrected_event("hello there"))
        .journal_at_once(user("item_1", "hello there"))
        .journal_at_once(interrupted(1))
        .journal_at_once(created(2))
        .journal(requested(2, &["item_1"]))
        .done()
}

/// The reply plays when the final comes: the message is rewritten, the reply plays on.
fn corrected_while_the_reply_plays() -> Recording {
    playing()
        .sent(said("hello there", true))
        .journal(corrected_event("hello there"))
        .journal_at_once(user("item_1", "hello there"))
        .later(5_000)
        .journal(Step::Emitted(SessionEvent::SpeechDone {
            speech: SpeechId(1),
            outcome: SpeechOutcome::Completed,
        }))
        .storage(settled(false))
        .storage(settled(true))
        .storage(saved("hello there"))
        .done()
}

fn playing() -> Authored {
    committed().journal(Step::Speech(SpeechStep::AgentStartedSpeaking { speech: SpeechId(1) }))
}

fn settled(answered: bool) -> StorageStep {
    let (speech, operation) = (1, StorageOperation::Settle);
    if answered {
        StorageStep::Answered { speech, operation, result: Ok(()), delivered: true }
    } else {
        StorageStep::Requested { speech, operation }
    }
}

fn saved(text: &str) -> StorageStep {
    StorageStep::Saved {
        items: vec![ChatItem::Message(lk_agents_abi::ChatMessage::text(
            "item_1", Role::User, text, std::time::UNIX_EPOCH,
        ))],
    }
}

/// The history reads the full words, the saved thread still the words they replaced.
fn correction_not_saved() -> Recording {
    playing()
        .sent(said("hello there", true))
        .journal(corrected_event("hello there"))
        .journal_at_once(user("item_1", "hello there"))
        .later(5_000)
        .storage(settled(false))
        .storage(settled(true))
        .storage(saved("hello"))
        .done()
}

/// The late final never reaches the message.
fn correction_dropped() -> Recording {
    playing().sent(said("hello there", true)).later(5_000).done()
}

/// An empty final took back the words the turn committed with.
fn empty_final_corrected() -> Recording {
    playing()
        .sent(said("", true))
        .journal(corrected_event(""))
        .journal_at_once(user("item_1", ""))
        .done()
}

/// The reply was heard, and the correction cut it to make it again.
fn restarted_after_playback() -> Recording {
    playing()
        .sent(said("hello there", true))
        .journal(corrected_event("hello there"))
        .journal_at_once(user("item_1", "hello there"))
        .journal_at_once(interrupted(1))
        .journal_at_once(created(2))
        .done()
}
