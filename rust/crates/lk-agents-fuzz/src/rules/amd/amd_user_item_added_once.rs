use lk_agents_voice::agent::Source;
use lk_agents_voice::amd::{AmdAnswer, AmdRunId};
use lk_agents_voice::audio_recognition::TurnId;
use lk_agents_voice::events::SessionEvent;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Observed, Recording},
};

// Spec §7.2: a realtime model without its own user transcription and without session STT gets
// the turn's words from AMD, which inserts them into the conversation and emits
// `conversation_item_added`. The proposal (Realtime) sends `InsertUserItem` only then, "so the
// item is never added twice". Judged at the host: each insertion the session took while its run
// was attached adds exactly one user item with those words, and the turn has no user item of
// its own words besides it. An insertion after `Detached` belongs to no attached run.

pub struct AmdUserItemAddedOnce;

impl Rule for AmdUserItemAddedOnce {
    const CONTRACT: Contract = Contract {
        id: "amd_user_item_added_once",
        guarantee: "A user transcript AMD inserts reaches the conversation exactly once, and \
                    never beside the turn's own words.",
        examples: &[inserted_once, inserted_then_typed],
        counterexamples: &[inserted_but_dropped, inserted_twice, inserted_beside_the_words],
        probe: Some(an_untranscribed_turn),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        let inserted = amd.inserted_items();
        let events = r.host_view().events();
        let users: Vec<(Observed, String, String, &Source)> = events
            .iter()
            .filter_map(|(at, e)| match e {
                SessionEvent::ConversationItemAdded { item, source, .. } => {
                    let m = item.as_message()?;
                    (m.role == lk_agents_abi::Role::User).then(|| {
                        (*at, m.id.clone(), m.raw_text_content().unwrap_or_default(), source)
                    })
                }
                _ => None,
            })
            .collect();
        for (i, (at, run, turn, text)) in inserted.iter().enumerate() {
            let attached = amd.attachments().into_iter().find(|a| a.run == *run);
            if attached.is_none_or(|a| *at >= a.until()) {
                continue;
            }
            let until = inserted.get(i + 1).map_or(Observed::END, |next| next.0);
            // Text the host typed, or another turn's words, is not this turn's item even with the
            // same words (`detector.py` `_add_realtime_transcript` inserts one item for the turn).
            let for_turn = |source: &Source| match source {
                Source::UserTyped => false,
                Source::UserTurn { turn: TurnId(t) } => t == turn,
                _ => true,
            };
            let added: Vec<_> = users
                .iter()
                .filter(|(seen, _, t, source)| {
                    seen > at && *seen < until && t == text && for_turn(source)
                })
                .collect();
            check.expect_with(Subject::Session, added.len() == 1, || {
                format!(
                    "AMD inserted {text:?} for turn {turn} at {} ms; the conversation added it \
                     {} times",
                    at.at_ms,
                    added.len()
                )
            });
            let ids: Vec<&String> = added.iter().map(|(_, id, ..)| id).collect();
            let own = users.iter().any(|(_, id, t, source)| {
                matches!(source, Source::UserTurn { turn: TurnId(t) } if t == turn)
                    && !t.trim().is_empty()
                    && !ids.contains(&id)
            });
            check.expect_with(Subject::Session, !own, || {
                format!("turn {turn} kept its own words beside the {text:?} AMD inserted")
            });
        }
    }
}

const WORDS: &str = "Please state your name.";

fn turn() -> Authored {
    Authored::new()
        .amd_entered(1)
        .journal(lk_agents_voice::trace::Step::Gate(lk_agents_voice::trace::GateStep::Attached {
            run: AmdRunId(1),
        }))
        .user_turn(None, 1, "")
}

fn insert(a: Authored) -> Authored {
    a.amd_answer(AmdAnswer::InsertUserItem { run: AmdRunId(1), turn: TurnId(1), text: WORDS.into() })
}

fn item(a: Authored, id: &str, text: &str, source: Source) -> Authored {
    a.session(SessionEvent::ConversationItemAdded {
        item: Box::new(lk_agents_abi::ChatItem::Message(lk_agents_abi::ChatMessage::text(
            id,
            lk_agents_abi::Role::User,
            text,
            std::time::UNIX_EPOCH,
        ))),
        agent: "agent".into(),
        source,
        visibility: lk_agents_voice::agent::Visibility::Both,
    })
}

fn inserted_once() -> Recording {
    item(insert(turn()), "amd_1", WORDS, Source::HostCtx).done()
}

/// The host later types the same words: its own item, not a second copy.
fn inserted_then_typed() -> Recording {
    let a = item(insert(turn()), "amd_1", WORDS, Source::HostCtx);
    item(a, "typed_1", WORDS, Source::UserTyped).done()
}

/// The gate drops the insertion, as a realtime session refused at attach would.
fn inserted_but_dropped() -> Recording {
    insert(turn()).done()
}

fn inserted_twice() -> Recording {
    let a = item(insert(turn()), "amd_1", WORDS, Source::HostCtx);
    item(a, "amd_2", WORDS, Source::HostCtx).done()
}

/// The session saved the recogniser's words for the turn, and AMD inserted its own.
fn inserted_beside_the_words() -> Recording {
    let a = item(turn(), "user_1b", "Please state your name", Source::UserTurn { turn: TurnId(1) });
    item(insert(a), "amd_1", WORDS, Source::HostCtx).done()
}

/// A realtime model without its own transcription and a session without STT: AMD's recogniser
/// hears the turn, and its words become the user item. Python:
/// `test_external_stt_preserves_history_without_native_transcription[amd-human]`.
fn an_untranscribed_turn() -> crate::drive::Case {
    use crate::drive::{Case, RealtimeSetup, RoomLegs, Setup};
    use crate::fakes::amd::AmdSetup;
    use crate::input::{AmdSays, Count, Input, Millis, Ref, UserText};
    let realtime = RealtimeSetup {
        turn_detection: false,
        tool_calls: Some(0),
        user_transcription: false,
        ..RealtimeSetup::default()
    };
    let setup = Setup {
        vad: true,
        realtime: Some(realtime),
        room_legs: Some(RoomLegs { pre_connect: None }),
        amd: Some(AmdSetup { own_stt: true, ..AmdSetup::default() }),
        ..Setup::default()
    };
    let advance = |ms| Input::Advance { ms: Millis(ms) };
    let inputs = vec![
        Input::AmdEnter {},
        advance(500),
        Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) },
        Input::SttStart { stt: Ref::new(0) },
        Input::MicFrames { mic: Ref::new(0), chunks: Count(30) },
        advance(300),
        Input::SttFinal { utterance: Ref::new(0), text: UserText(WORDS.into()), speaker: None },
        Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) },
        Input::SttEnd { utterance: Ref::new(0) },
        advance(600),
        Input::AmdReply { ask: Ref::new(0), says: AmdSays::Human },
        advance(4_000),
    ];
    Case { setup, inputs, ..Case::default() }
}
